import { memoize } from "functools-kit";
import ClientAgent, { IAgent } from "src/client/ClientAgent";
import TYPES from "src/config/types";
import { inject } from "src/core/di";
import { TContextService } from "src/services/base/ContextService";
import LoggerService from "src/services/base/LoggerService";
import NavigationRegistryService from "src/services/function/NavigationRegistryService";

const AGENT_PROMPT = `You are a sales agent that handles all actions related to placing an order to purchase an item.
Regardless of what the user wants to purchase, must ask for BOTH the user ID and product ID to place an order.
An order cannot be placed without these two pieces of information. Ask for both user_id and product_id in one message.
If the user asks you to notify them, you must ask them what their preferred method is. For notifications, you must
ask them for user_id and method in one message.
`;

export class SalesAgentService implements IAgent {
  readonly contextService = inject<TContextService>(TYPES.contextService);
  readonly loggerService = inject<LoggerService>(TYPES.loggerService);

  readonly navigationRegistryService = inject<NavigationRegistryService>(TYPES.navigationRegistryService);

  private getClientAgent = memoize(
    ([clientId]) => clientId,
    (clientId: string) =>
      new ClientAgent({
        clientId,
        agentName: "sales-agent",
        prompt: AGENT_PROMPT,
        tools: [
          this.navigationRegistryService.useNavigateToTriage(),
        ]
      })
  );

  public execute = async (input: string[]) => {
    this.loggerService.logCtx("salesAgentService execute", {
      input,
    });
    const agent = this.getClientAgent(this.contextService.context.clientId);
    return await agent.execute(input);
  };

  public beginChat = async () => {
    this.loggerService.logCtx("salesAgentService beginChat");
    const agent = this.getClientAgent(this.contextService.context.clientId);
    return await agent.beginChat();
  };

  commitSystemMessage = async (message: string) => {
    this.loggerService.logCtx("salesAgentService commitSystemMessage", {
      message,
    });
    const agent = this.getClientAgent(this.contextService.context.clientId);
    return await agent.commitSystemMessage(message);
  };

  commitToolOutput = async (content: string) => {
    this.loggerService.logCtx("salesAgentService commitToolOutput", {
      content,
    });
    const agent = this.getClientAgent(this.contextService.context.clientId);
    return await agent.commitToolOutput(content);
  };

  public dispose = async () => {
    this.loggerService.logCtx("salesAgentService dispose");
    const agent = this.getClientAgent(this.contextService.context.clientId);
    await agent.dispose();
    this.getClientAgent.clear(this.contextService.context.clientId);
  };
}

export default SalesAgentService;
