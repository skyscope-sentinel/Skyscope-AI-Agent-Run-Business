import ContextService from "../base/ContextService";
import { inject } from "src/core/di";
import TYPES from "src/config/types";
import LoggerService from "../base/LoggerService";
import { IConnection } from "src/client/ClientConnection";
import ConnectionPrivateService from "../private/ConnectionPrivateService";
import { AgentName } from "src/utils/getAgentMap";
import { IIncomingMessage, IOutgoingMessage } from "src/model/WsMessage.model";

type TConnection = {
  [key in keyof IConnection]: any;
};

export class ConnectionPublicService implements TConnection {
  readonly loggerService = inject<LoggerService>(TYPES.loggerService);
  readonly connectionPrivateService = inject<ConnectionPrivateService>(TYPES.connectionPrivateService);

  public execute = async (clientId: string, messages: string[], agentName: AgentName) => {
    this.loggerService.log("connectionPublicService execute", { clientId, messages, agentName });
    return await ContextService.runInContext(
      async () => {
        return await this.connectionPrivateService.execute(messages, agentName);
      },
      {
        clientId,
      }
    );
  };

  public waitForOutput = async (clientId: string) => {
    this.loggerService.log("connectionPublicService waitForOutput", { clientId });
    return await ContextService.runInContext(
      async () => {
        return await this.connectionPrivateService.waitForOutput();
      },
      {
        clientId,
      }
    );
  };

  public complete = async (clientId: string, messages: string[]) => {
    this.loggerService.log("connectionPublicService complete", { clientId, messages });
    return await ContextService.runInContext(
      async () => {
        return await this.connectionPrivateService.complete(messages);
      },
      {
        clientId,
      }
    );
  };

  public connect = (
    clientId: string,
    connector: (outgoing: IOutgoingMessage) => void | Promise<void>
  ) => {
    this.loggerService.log("connectionPublicService connect", { clientId });
    return ContextService.runInContext(
      () => {
        const receive = this.connectionPrivateService.connect(async (outgoing) => {
          return await ContextService.runInContext(
            async () => {
              return await connector(outgoing);
            },
            {
              clientId,
            }
          );
        });
        return async (incoming: IIncomingMessage) => {
          return await ContextService.runInContext(async () => {
            return await receive(incoming);
          }, {
            clientId,
          })
        }
      },
      {
        clientId,
      }
    );
  };

  public commitToolOutput = async (clientId: string, content: string, agentName: AgentName) => {
    this.loggerService.logCtx("connectionPublicService commitToolOutput", {
      content,
      agentName,
    });
    return await ContextService.runInContext(
      async () => {
        return await this.connectionPrivateService.commitToolOutput(content, agentName);
      },
      {
        clientId,
      }
    );
  };

  public commitSystemMessage = async (clientId: string, message: string, agentName: AgentName) => {
    this.loggerService.logCtx("connectionPublicService commitSystemMessage", {
      message,
      agentName,
    });
    return await ContextService.runInContext(
      async () => {
        return await this.connectionPrivateService.commitSystemMessage(message, agentName);
      },
      {
        clientId,
      }
    );
  };

  public dispose = async (clientId: string) => {
    this.loggerService.log("connectionPublicService dispose", { clientId });
    return await ContextService.runInContext(
      async () => {
        return await this.connectionPrivateService.dispose();
      },
      {
        clientId,
      }
    );
  };

  public emit = async (clientId: string, outgoing: string, agentName: AgentName) => {
    this.loggerService.log("connectionPublicService emit", { clientId, outgoing, agentName });
    return await ContextService.runInContext(
      async () => {
        return await this.connectionPrivateService.emit(outgoing, agentName);
      },
      {
        clientId,
      }
    );
  };
}

export default ConnectionPublicService;
