import { isSupportedModel, isVisionModel } from '@renderer/config/models'
import { SUMMARIZE_PROMPT } from '@renderer/config/prompts'
import { getAssistantSettings, getDefaultModel, getTopNamingModel } from '@renderer/services/assistant'
import { EVENT_NAMES } from '@renderer/services/event'
import { filterContextMessages } from '@renderer/services/messages'
import { Assistant, FileTypes, Message, Model, Provider, Suggestion } from '@renderer/types'
import { removeQuotes } from '@renderer/utils'
import { takeRight } from 'lodash'
import OpenAI, { AzureOpenAI } from 'openai'
import {
  ChatCompletionContentPart,
  ChatCompletionCreateParamsNonStreaming,
  ChatCompletionMessageParam
} from 'openai/resources'

import BaseProvider from './BaseProvider'

export default class OpenAIProvider extends BaseProvider {
  private sdk: OpenAI

  constructor(provider: Provider) {
    super(provider)
    if (provider.id === 'azure-openai') {
      this.sdk = new AzureOpenAI({
        dangerouslyAllowBrowser: true,
        apiKey: provider.apiKey,
        apiVersion: provider.apiVersion,
        endpoint: provider.apiHost
      })
      return
    }

    this.sdk = new OpenAI({
      dangerouslyAllowBrowser: true,
      apiKey: provider.apiKey,
      baseURL: this.getBaseURL()
    })
  }

  private isSupportStreamOutput(modelId: string): boolean {
    if (modelId.includes('o1-')) {
      return false
    }
    return true
  }

  private get isNotSupportFiles() {
    const providers = ['deepseek', 'baichuan', 'minimax', 'doubao']
    return providers.includes(this.provider.id)
  }

  private async getMessageParam(
    message: Message,
    model: Model
  ): Promise<OpenAI.Chat.Completions.ChatCompletionMessageParam> {
    const isVision = isVisionModel(model)

    if (!message.files) {
      return {
        role: message.role,
        content: message.content
      }
    }

    if (this.isNotSupportFiles) {
      if (message.files) {
        const textFiles = message.files.filter((file) => [FileTypes.TEXT, FileTypes.DOCUMENT].includes(file.type))

        if (textFiles.length > 0) {
          let text = ''
          const divider = '\n\n---\n\n'

          for (const file of textFiles) {
            const fileContent = (await window.api.file.read(file.id + file.ext)).trim()
            const fileNameRow = 'file: ' + file.origin_name + '\n\n'
            text = text + fileNameRow + fileContent + divider
          }

          return {
            role: message.role,
            content: message.content + divider + text
          }
        }
      }

      return {
        role: message.role,
        content: message.content
      }
    }

    const parts: ChatCompletionContentPart[] = [
      {
        type: 'text',
        text: message.content
      }
    ]

    for (const file of message.files || []) {
      if (file.type === FileTypes.IMAGE && isVision) {
        const image = await window.api.file.base64Image(file.id + file.ext)
        parts.push({
          type: 'image_url',
          image_url: { url: image.data }
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
    } as ChatCompletionMessageParam
  }

  async completions({ messages, assistant, onChunk, onFilterMessages }: CompletionsParams): Promise<void> {
    const defaultModel = getDefaultModel()
    const model = assistant.model || defaultModel
    const { contextCount, maxTokens, streamOutput } = getAssistantSettings(assistant)

    const systemMessage = assistant.prompt ? { role: 'system', content: assistant.prompt } : undefined
    const userMessages: ChatCompletionMessageParam[] = []

    const _messages = filterContextMessages(takeRight(messages, contextCount + 1))
    onFilterMessages(_messages)

    for (const message of _messages) {
      userMessages.push(await this.getMessageParam(message, model))
    }

    const isOpenAIo1 = model.id.includes('o1-')
    const isSupportStreamOutput = streamOutput && this.isSupportStreamOutput(model.id)

    // @ts-ignore key is not typed
    const stream = await this.sdk.chat.completions.create({
      model: model.id,
      messages: [isOpenAIo1 ? undefined : systemMessage, ...userMessages].filter(
        Boolean
      ) as ChatCompletionMessageParam[],
      temperature: isOpenAIo1 ? 1 : assistant?.settings?.temperature,
      max_tokens: maxTokens,
      keep_alive: this.keepAliveTime,
      stream: isSupportStreamOutput
    })

    if (!isSupportStreamOutput) {
      return onChunk({
        text: stream.choices[0].message?.content || '',
        usage: stream.usage
      })
    }

    for await (const chunk of stream) {
      if (window.keyv.get(EVENT_NAMES.CHAT_COMPLETION_PAUSED)) {
        break
      }

      onChunk({
        text: chunk.choices[0]?.delta?.content || '',
        usage: chunk.usage
      })
    }
  }

  async translate(message: Message, assistant: Assistant) {
    const defaultModel = getDefaultModel()
    const model = assistant.model || defaultModel
    const messages = [
      { role: 'system', content: assistant.prompt },
      { role: 'user', content: message.content }
    ]

    // @ts-ignore key is not typed
    const response = await this.sdk.chat.completions.create({
      model: model.id,
      messages: messages as ChatCompletionMessageParam[],
      stream: false,
      keep_alive: this.keepAliveTime
    })

    return response.choices[0].message?.content || ''
  }

  public async summaries(messages: Message[], assistant: Assistant): Promise<string> {
    const model = getTopNamingModel() || assistant.model || getDefaultModel()

    const userMessages = takeRight(messages, 5)
      .filter((message) => !message.isPreset)
      .map((message) => ({
        role: message.role,
        content: message.content
      }))

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

    // @ts-ignore key is not typed
    const response = await this.sdk.chat.completions.create({
      model: model.id,
      messages: [systemMessage, userMessage] as ChatCompletionMessageParam[],
      stream: false,
      keep_alive: this.keepAliveTime,
      max_tokens: 1000
    })

    return removeQuotes(response.choices[0].message?.content?.substring(0, 50) || '')
  }

  public async generateText({ prompt, content }: { prompt: string; content: string }): Promise<string> {
    const model = getDefaultModel()

    const response = await this.sdk.chat.completions.create({
      model: model.id,
      stream: false,
      messages: [
        { role: 'system', content: prompt },
        { role: 'user', content }
      ]
    })

    return response.choices[0].message?.content || ''
  }

  async suggestions(messages: Message[], assistant: Assistant): Promise<Suggestion[]> {
    const model = assistant.model

    if (!model) {
      return []
    }

    const response: any = await this.sdk.request({
      method: 'post',
      path: '/advice_questions',
      body: {
        messages: messages.filter((m) => m.role === 'user').map((m) => ({ role: m.role, content: m.content })),
        model: model.id,
        max_tokens: 0,
        temperature: 0,
        n: 0
      }
    })

    return response?.questions?.filter(Boolean)?.map((q: any) => ({ content: q })) || []
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
      const response = await this.sdk.chat.completions.create(body as ChatCompletionCreateParamsNonStreaming)

      return {
        valid: Boolean(response?.choices[0].message),
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
    try {
      const query: Record<string, any> = {}

      if (this.provider.id === 'silicon') {
        query.type = 'text'
      }

      const response = await this.sdk.models.list({ query })

      if (this.provider.id === 'github') {
        // @ts-ignore key is not typed
        return response.body
          .map((model) => ({
            id: model.name,
            description: model.summary,
            object: 'model',
            owned_by: model.publisher
          }))
          .filter(isSupportedModel)
      }

      if (this.provider.id === 'together') {
        // @ts-ignore key is not typed
        return response?.body
          .map((model: any) => ({
            id: model.id,
            description: model.display_name,
            object: 'model',
            owned_by: model.organization
          }))
          .filter(isSupportedModel)
      }

      const models = response?.data || []

      return models.filter(isSupportedModel)
    } catch (error) {
      return []
    }
  }
}
