import { not, or, queued, Subject } from "functools-kit";
import { omit } from "lodash-es";
import { Message, Tool } from "ollama";
import { CC_OLLAMA_EMIT_TOOL_PROTOCOL } from "src/config/params";
import TYPES from "src/config/types";
import { inject } from "src/core/di";
import LoggerService from "src/services/base/LoggerService";
import { AgentName } from "../utils/getAgentMap";
import ConnectionPrivateService from "src/services/private/ConnectionPrivateService";
import CompletionService from "src/services/api/CompletionService";
import { TContextService } from "src/services/base/ContextService";
import validateNoToolCall from "src/validation/validateNoToolCall";
import { IModelMessage } from "src/model/ModelMessage.model";
import ClientHistoryDbService from "src/services/db/ClientHistoryDbService";

/**
 * @see https://github.com/ollama/ollama/blob/86a622cbdc69e9fd501764ff7565e977fc98f00a/server/model.go#L158
 */
const TOOL_PROTOCOL_PROMPT = `For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
`;

export interface IAgentTool<T = Record<string, unknown>> {
  call(agentName: AgentName, params: T): Promise<void>;
  validate(agentName: AgentName, params: T): Promise<boolean> | boolean;
  getToolSignature(): AgentToolSignature<T>;
}

export interface AgentToolSignature<T = Record<string, unknown>>
  extends Tool,
    Omit<
      IAgentTool<T>,
      keyof {
        call: never;
        getToolSignature: never;
      }
    > {
  implementation(agentName: AgentName, params: T): Promise<void>;
}

interface IAgentParams {
  agentName: AgentName;
  clientId: string;
  prompt: string;
  tools?: AgentToolSignature[];
}

export interface IAgent {
  execute: (input: string[]) => Promise<void>;
  beginChat: () => Promise<void>;
  commitToolOutput(content: string): Promise<void>;
  commitSystemMessage(message: string): Promise<void>;
}

/**
 * @description `ask for agent function` in `llama3.1:8b` to troubleshoot (need CC_OLLAMA_EMIT_TOOL_PROTOCOL to be turned off)
 */
const TOOL_CALL_EXCEPTION_PROMPT = "Start the conversation";

/**
 * @description When the model output is empty just say hello to the customer
 */
const EMPTY_OUTPUT_PLACEHOLDERS = [
  "Sorry, I missed that. Could you say it again?",
  "I couldn't catch that. Would you mind repeating?",
  "I didn’t quite hear you. Can you repeat that, please?",
  "Pardon me, I didn’t hear that clearly. Could you repeat it?",
  "Sorry, I didn’t catch what you said. Could you say it again?",
  "Could you repeat that? I didn’t hear it clearly.",
  "I missed that. Can you say it one more time?",
  "Sorry, I didn’t get that. Could you repeat it, please?",
  "I didn’t hear you properly. Can you say that again?",
  "Could you please repeat that? I didn’t catch it.",
];

const getPlaceholder = () =>
  EMPTY_OUTPUT_PLACEHOLDERS[
    Math.floor(Math.random() * EMPTY_OUTPUT_PLACEHOLDERS.length)
  ];

export class ClientAgent implements IAgent {
  readonly loggerService = inject<LoggerService>(TYPES.loggerService);
  readonly contextService = inject<TContextService>(TYPES.contextService);
  readonly completionService = inject<CompletionService>(
    TYPES.completionService
  );
  readonly connectionPrivateService = inject<ConnectionPrivateService>(
    TYPES.connectionPrivateService
  );

  readonly clientHistoryDbService = inject<ClientHistoryDbService>(
    TYPES.clientHistoryDbService
  );

  readonly _toolCommitSubject = new Subject<void>();

  constructor(readonly params: IAgentParams) {}

  _emitOuput = async (result: string) => {
    this.loggerService.debugCtx(
      `ClientAgent agentName=${this.params.agentName} _emitOuput`
    );
    if (!result) {
      const result = await this._resurrectModel();
      if (!result) {
        throw new Error(`clientAgent agentName=${this.params.agentName} model ressurect failed`);
      }
      await this.connectionPrivateService.emit(result, this.params.agentName);
      return;
    }
    await this.connectionPrivateService.emit(result, this.params.agentName);
    return;
  }

  _resurrectModel = async () => {
    this.loggerService.debugCtx(
      `ClientAgent agentName=${this.params.agentName} _resurrectModel`
    );
    {
      await this.clientHistoryDbService.clear(this.params.agentName);
      await this.beginChat();
      await this.clientHistoryDbService.push(this.params.agentName, {
        role: "user",
        agentName: this.params.agentName,
        content: TOOL_CALL_EXCEPTION_PROMPT,
      });
    }
    const response = await this.getCompletion();
    const result = response.message.content;
    if (!result) {
      this.loggerService.debugCtx(
        `ClientAgent agentName=${this.params.agentName} _resurrectModel empty output`
      );
      return getPlaceholder();
    }
    await this.clientHistoryDbService.push(this.params.agentName, {
      ...response.message,
      role: response.message.role as IModelMessage["role"],
      agentName: this.params.agentName,
    });
    return result;
  };

  getCompletion = async (): Promise<{ message: Message }> => {
    this.loggerService.debugCtx(
      `ClientAgent agentName=${this.params.agentName} getCompletion`
    );
    const messages = await this.clientHistoryDbService.toArrayForAgent(
      this.params.agentName
    );
    return await this.completionService.getCompletion(
      messages.map((message) => omit(message, "agentName")),
      this.params.tools?.map((t) => omit(t, "implementation"))
    );
  };

  beginChat = async () => {
    this.loggerService.debugCtx(
      `ClientAgent agentName=${this.params.agentName} beginChat`
    );
    await this.clientHistoryDbService.push(this.params.agentName, {
      role: "system",
      agentName: this.params.agentName,
      content: this.params.prompt.trim(),
    });
    if (CC_OLLAMA_EMIT_TOOL_PROTOCOL) {
      await this.clientHistoryDbService.push(this.params.agentName, {
        role: "system",
        agentName: this.params.agentName,
        content: TOOL_PROTOCOL_PROMPT,
      });
    }
  };

  commitSystemMessage = async (message: string): Promise<void> => {
    this.loggerService.debugCtx(
      `ClientAgent agentName=${this.params.agentName} commitSystemMessage`
    );
    await this.clientHistoryDbService.push(this.params.agentName, {
      role: "system",
      agentName: this.params.agentName,
      content: message.trim(),
    });
  };

  commitToolOutput = async (content: string): Promise<void> => {
    this.loggerService.debugCtx(
      `ClientAgent agentName=${this.params.agentName} commitToolOutput content=${content}`
    );
    await this.clientHistoryDbService.push(this.params.agentName, {
      role: "tool",
      agentName: this.params.agentName,
      content,
    });
    await this._toolCommitSubject.next();
  };

  execute = queued(async (messages: string[]) => {
    this.loggerService.debugCtx(
      `ClientAgent agentName=${this.params.agentName} execute begin`,
      { messages }
    );
    if (await not(this.clientHistoryDbService.length(this.params.agentName))) {
      await this.beginChat();
    }
    for (const message of messages) {
      await this.clientHistoryDbService.push(this.params.agentName, {
        role: "user",

        agentName: this.params.agentName,
        content: message.trim(),
      });
    }
    const response = await this.getCompletion();
    if (response.message.tool_calls) {
      this.loggerService.debugCtx(
        `ClientAgent agentName=${this.params.agentName} tool call begin`
      );
      for (const tool of response.message.tool_calls) {
        const targetFn = this.params.tools?.find(
          (t) => t.function.name === tool.function.name
        );
        await this.clientHistoryDbService.push(this.params.agentName, {
          ...response.message,
          role: response.message.role as IModelMessage["role"],
          agentName: this.params.agentName,
        });
        if (
          targetFn &&
          (await targetFn.validate(
            this.params.agentName,
            tool.function.arguments
          ))
        ) {
          await targetFn.implementation(
            this.params.agentName,
            tool.function.arguments
          );
          this.loggerService.debugCtx(
            `ClientAgent agentName=${this.params.agentName} functionName=${tool.function.name} tool call executing`
          );
          await Promise.race([
            this._toolCommitSubject.toPromise(),
            this.connectionPrivateService.waitForOutput(),
          ]);
          this.loggerService.debugCtx(
            `ClientAgent agentName=${this.params.agentName} functionName=${tool.function.name} tool call end`
          );
          return;
        }
        this.loggerService.debugCtx(
          `ClientAgent agentName=${this.params.agentName} functionName=${tool.function.name} tool function not found`
        );
        const result = await this._resurrectModel();
        this.loggerService.debugCtx(
          `ClientAgent agentName=${this.params.agentName} execute end result=${result}`
        );
        await this._emitOuput(result);
        return;
      }
    }
    if (!response.message.tool_calls) {
      this.loggerService.debugCtx(
        `ClientAgent agentName=${this.params.agentName} execute no tool calls detected`
      );
    }
    const result = response.message.content;
    await this.clientHistoryDbService.push(this.params.agentName, {
      ...response.message,
      role: response.message.role as IModelMessage["role"],
      agentName: this.params.agentName,
    });
    const isInvalid = await or(not(validateNoToolCall(result)), !result);
    isInvalid &&
      this.loggerService.debugCtx(
        `ClientAgent agentName=${this.params.agentName} execute invalid tool call detected`,
        { result }
      );
    if (isInvalid) {
      const result = await this._resurrectModel();
      await this._emitOuput(result);
      return;
    }
    this.loggerService.debugCtx(
      `ClientAgent agentName=${this.params.agentName} execute end result=${result}`
    );
    await this._emitOuput(result);
  }) as unknown as IAgent["execute"];

  dispose = async () => {
    this.loggerService.debugCtx(
      `ClientAgent agentName=${this.params.agentName} dispose`
    );
    await this.clientHistoryDbService.dispose(this.params.agentName);
  };
}

export default ClientAgent;
