import BaseProvider from '@renderer/providers/BaseProvider'
import ProviderFactory from '@renderer/providers/ProviderFactory'
import { Assistant, Message, Provider, Suggestion } from '@renderer/types'
import OpenAI from 'openai'

export default class AiProvider {
  private sdk: BaseProvider

  constructor(provider: Provider) {
    this.sdk = ProviderFactory.create(provider)
  }

  public async completions({ messages, assistant, onChunk, onFilterMessages }: CompletionsParams): Promise<void> {
    return this.sdk.completions({ messages, assistant, onChunk, onFilterMessages })
  }

  public async translate(message: Message, assistant: Assistant): Promise<string> {
    return this.sdk.translate(message, assistant)
  }

  public async summaries(messages: Message[], assistant: Assistant): Promise<string> {
    return this.sdk.summaries(messages, assistant)
  }

  public async suggestions(messages: Message[], assistant: Assistant): Promise<Suggestion[]> {
    return this.sdk.suggestions(messages, assistant)
  }

  public async generateText({ prompt, content }: { prompt: string; content: string }): Promise<string> {
    return this.sdk.generateText({ prompt, content })
  }

  public async check(): Promise<{ valid: boolean; error: Error | null }> {
    return this.sdk.check()
  }

  public async models(): Promise<OpenAI.Models.Model[]> {
    return this.sdk.models()
  }
}
