import Anthropic from '@anthropic-ai/sdk'
import { MessageCreateParamsNonStreaming, MessageParam } from '@anthropic-ai/sdk/resources'
import { DEFAULT_MAX_TOKENS } from '@renderer/config/constant'
import { SUMMARIZE_PROMPT } from '@renderer/config/prompts'
import { getAssistantSettings, getDefaultModel, getTopNamingModel } from '@renderer/services/assistant'
import { EVENT_NAMES } from '@renderer/services/event'
import { filterContextMessages } from '@renderer/services/messages'
import { Assistant, FileTypes, Message, Provider, Suggestion } from '@renderer/types'
import { first, flatten, sum, takeRight } from 'lodash'
import OpenAI from 'openai'

import BaseProvider from './BaseProvider'

export default class AnthropicProvider extends BaseProvider {
  private sdk: Anthropic

  constructor(provider: Provider) {
    super(provider)
    this.sdk = new Anthropic({ apiKey: provider.apiKey, baseURL: this.getBaseURL() })
  }

  public getBaseURL(): string {
    return this.provider.apiHost
  }

  private async getMessageParam(message: Message): Promise<MessageParam> {
    const parts: MessageParam['content'] = [{ type: 'text', text: message.content }]

    for (const file of message.files || []) {
      if (file.type === FileTypes.IMAGE) {
        const base64Data = await window.api.file.base64Image(file.id + file.ext)
        parts.push({
          type: 'image',
          source: {
            data: base64Data.base64,
            media_type: base64Data.mime.replace('jpg', 'jpeg') as any,
            type: 'base64'
          }
        })
      }
      if ([FileTypes.TEXT, FileTypes.DOCUMENT].includes(file.type)) {
        const fileContent = await (await window.api.file.read(file.id + file.ext)).trim()
        parts.push({
          type: 'text',
          text: file.origin_name + '\n' + fileContent
        })
      }
    }

    return {
      role: message.role,
      content: parts
    }
  }

  public async completions({ messages, assistant, onChunk, onFilterMessages }: CompletionsParams) {
    const defaultModel = getDefaultModel()
    const model = assistant.model || defaultModel
    const { contextCount, maxTokens, streamOutput } = getAssistantSettings(assistant)

    const userMessagesParams: MessageParam[] = []
    const _messages = filterContextMessages(takeRight(messages, contextCount + 2))

    onFilterMessages(_messages)

    for (const message of _messages) {
      userMessagesParams.push(await this.getMessageParam(message))
    }

    const userMessages = flatten(userMessagesParams)

    if (first(userMessages)?.role === 'assistant') {
      userMessages.shift()
    }

    const body: MessageCreateParamsNonStreaming = {
      model: model.id,
      messages: userMessages,
      max_tokens: maxTokens || DEFAULT_MAX_TOKENS,
      temperature: assistant?.settings?.temperature,
      system: assistant.prompt
    }

    if (!streamOutput) {
      const message = await this.sdk.messages.create({ ...body, stream: false })
      return onChunk({
        text: message.content[0].type === 'text' ? message.content[0].text : '',
        usage: message.usage
      })
    }

    return new Promise<void>((resolve, reject) => {
      const stream = this.sdk.messages
        .stream({ ...body, stream: true })
        .on('text', (text) => {
          if (window.keyv.get(EVENT_NAMES.CHAT_COMPLETION_PAUSED)) {
            stream.controller.abort()
            return resolve()
          }
          onChunk({ text })
        })
        .on('finalMessage', (message) => {
          onChunk({
            text: '',
            usage: {
              prompt_tokens: message.usage.input_tokens,
              completion_tokens: message.usage.output_tokens,
              total_tokens: sum(Object.values(message.usage))
            }
          })
          resolve()
        })
        .on('error', (error) => reject(error))
    })
  }

  public async translate(message: Message, assistant: Assistant) {
    const defaultModel = getDefaultModel()
    const model = assistant.model || defaultModel
    const messages = [
      { role: 'system', content: assistant.prompt },
      { role: 'user', content: message.content }
    ]

    const response = await this.sdk.messages.create({
      model: model.id,
      messages: messages.filter((m) => m.role === 'user') as MessageParam[],
      max_tokens: 4096,
      temperature: assistant?.settings?.temperature,
      system: assistant.prompt,
      stream: false
    })

    return response.content[0].type === 'text' ? response.content[0].text : ''
  }

  public async summaries(messages: Message[], assistant: Assistant): Promise<string> {
    const model = getTopNamingModel() || assistant.model || getDefaultModel()

    const userMessages = takeRight(messages, 5)
      .filter((message) => !message.isPreset)
      .map((message) => ({
        role: message.role,
        content: message.content
      }))

    if (first(userMessages)?.role === 'assistant') {
      userMessages.shift()
    }

    const userMessageContent = userMessages.reduce((prev, curr) => {
      const content = curr.role === 'user' ? `User: ${curr.content}` : `Assistant: ${curr.content}`
      return prev + (prev ? '\n' : '') + content
    }, '')

    const systemMessage = {
      role: 'system',
      content: SUMMARIZE_PROMPT
    }

    const userMessage = {
      role: 'user',
      content: userMessageContent
    }

    const message = await this.sdk.messages.create({
      messages: [userMessage] as Anthropic.Messages.MessageParam[],
      model: model.id,
      system: systemMessage.content,
      stream: false,
      max_tokens: 4096
    })

    return message.content[0].type === 'text' ? message.content[0].text : ''
  }

  public async generateText({ prompt, content }: { prompt: string; content: string }): Promise<string> {
    const model = getDefaultModel()

    const message = await this.sdk.messages.create({
      model: model.id,
      system: prompt,
      stream: false,
      max_tokens: 4096,
      messages: [
        {
          role: 'user',
          content
        }
      ]
    })

    return message.content[0].type === 'text' ? message.content[0].text : ''
  }

  public async suggestions(): Promise<Suggestion[]> {
    return []
  }

  public async check(): Promise<{ valid: boolean; error: Error | null }> {
    const model = this.provider.models[0]

    const body = {
      model: model.id,
      messages: [{ role: 'user', content: 'hi' }],
      max_tokens: 100,
      stream: false
    }

    try {
      const message = await this.sdk.messages.create(body as MessageCreateParamsNonStreaming)
      return {
        valid: message.content.length > 0,
        error: null
      }
    } catch (error: any) {
      return {
        valid: false,
        error
      }
    }
  }

  public async models(): Promise<OpenAI.Models.Model[]> {
    return []
  }
}
