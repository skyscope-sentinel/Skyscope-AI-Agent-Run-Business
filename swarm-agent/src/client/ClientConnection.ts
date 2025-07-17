import { Subject } from "functools-kit";
import TYPES from "src/config/types";
import { inject } from "src/core/di";
import LoggerService from "src/services/base/LoggerService";
import getAgentMap, { AgentName } from "../utils/getAgentMap";
import RootSwarmService from "src/services/logic/RootSwarmService";
import { IOutgoingMessage, IIncomingMessage } from "src/model/WsMessage.model";

interface IConnectionParams {
  connectionName: string;
  clientId: string;
}

type SendMessageFn = (outgoing: IOutgoingMessage) => Promise<void>;
type ReceiveMessageFn = (incoming: IIncomingMessage) => Promise<void>;

export interface IConnection {
  dispose(): Promise<void>;
  waitForOutput(): Promise<string>;
  connect(connector: SendMessageFn): ReceiveMessageFn;
  commitToolOutput(content: string, agentName: AgentName): Promise<void>;
  commitSystemMessage(message: string, agentName: AgentName): Promise<void>;
  emit(outgoing: string, agentName: AgentName): Promise<void>;
}

export class BaseConnection {
  readonly rootSwarmService = inject<RootSwarmService>(TYPES.rootSwarmService);

  readonly loggerService = inject<LoggerService>(TYPES.loggerService);

  readonly outgoingSubject = new Subject<IOutgoingMessage>();

  constructor(readonly params: IConnectionParams) {}

  waitForOutput = async (): Promise<string> => {
    this.loggerService.debugCtx("BaseConnection waitForOutput", {
      connectionName: this.params.connectionName,
    });
    const { data } = await this.outgoingSubject.toPromise();
    return data;
  };

  execute = async (message: string[], agentName: AgentName) => {
    this.loggerService.debugCtx("BaseConnection execute", {
      message,
      connectionName: this.params.connectionName,
    });
    if ((await this.rootSwarmService.getAgentName()) !== agentName) {
      return;
    }
    const agent = await this.rootSwarmService.getAgent();
    return await agent.execute(message);
  };

  commitToolOutput = async (content: string, agentName: AgentName) => {
    this.loggerService.debugCtx("BaseConnection commitToolOutput", {
      content,
      connectionName: this.params.connectionName,
      agentName,
    });
    if ((await this.rootSwarmService.getAgentName()) !== agentName) {
      return;
    }
    const agent = await this.rootSwarmService.getAgent();
    return await agent.commitToolOutput(content);
  };

  commitSystemMessage = async (message: string, agentName: AgentName) => {
    this.loggerService.debugCtx("BaseConnection commitSystemMessage", {
      message,
      connectionName: this.params.connectionName,
      agentName,
    });
    if ((await this.rootSwarmService.getAgentName()) !== agentName) {
      return;
    }
    const agent = await this.rootSwarmService.getAgent();
    return await agent.commitSystemMessage(message);
  };

  connect = (
    connector: (outgoing: IOutgoingMessage) => void | Promise<void>
  ) => {
    this.loggerService.debugCtx("BaseConnection connect", {
      connectionName: this.params.connectionName,
    });
    this.outgoingSubject.subscribe(async (outgoing) => {
      await connector(outgoing);
    });
    return async (incoming: IIncomingMessage) => {
      this.loggerService.debugCtx("BaseConnection connect call", {
        connectionName: this.params.connectionName,
      });
      await this.execute(
        incoming.data,
        await this.rootSwarmService.getAgentName()
      );
    };
  };

  emit = async (outgoing: string, agentName: AgentName) => {
    this.loggerService.debugCtx("BaseConnection emit", {
      connectionName: this.params.connectionName,
      agentName,
    });
    if ((await this.rootSwarmService.getAgentName()) !== agentName) {
      return;
    }
    await this.outgoingSubject.next({
      clientId: this.params.clientId,
      stamp: Date.now().toString(),
      agentName,
      data: outgoing,
    });
  };

  dispose = async () => {
    this.loggerService.debugCtx("BaseConnection dispose", {
      connectionName: this.params.connectionName,
    });
    for (const agent of Object.values(getAgentMap())) {
      await agent.dispose();
    }
    await this.rootSwarmService.dispose();
    this.outgoingSubject.unsubscribeAll();
  };
}

export default BaseConnection;
