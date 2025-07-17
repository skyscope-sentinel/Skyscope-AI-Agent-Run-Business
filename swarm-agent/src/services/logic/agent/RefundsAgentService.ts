import { memoize } from "functools-kit";
import ClientAgent, { IAgent } from "src/client/ClientAgent";
import TYPES from "src/config/types";
import { inject } from "src/core/di";
import { TContextService } from "src/services/base/ContextService";
import LoggerService from "src/services/base/LoggerService";
import NavigationRegistryService from "src/services/function/NavigationRegistryService";

const AGENT_PROMPT = `You are a refund agent that handles all actions related to refunds after a return has been processed.
You must ask for both the user ID and item ID to initiate a refund. Ask for both user_id and item_id in one message.
If the user asks you to notify them, you must ask them what their preferred method of notification is. For notifications, you must
ask them for user_id and method in one message
`;

export class RefundsAgentService implements IAgent {
  readonly contextService = inject<TContextService>(TYPES.contextService);
  readonly loggerService = inject<LoggerService>(TYPES.loggerService);

  readonly navigationRegistryService = inject<NavigationRegistryService>(TYPES.navigationRegistryService);

  private getClientAgent = memoize(
    ([clientId]) => clientId,
    (clientId: string) =>
      new ClientAgent({
        clientId,
        agentName: "refunds-agent",
        prompt: AGENT_PROMPT,
        tools: [
          this.navigationRegistryService.useNavigateToTriage(),
        ]
      })
  );

  public commitSystemMessage = async (message: string) => {
    this.loggerService.logCtx("refundsAgentService commitSystemMessage", {
      message,
    });
    const agent = this.getClientAgent(this.contextService.context.clientId);
    return await agent.commitSystemMessage(message);
  };

  public commitToolOutput = async (content: string) => {
    this.loggerService.logCtx("refundsAgentService commitToolOutput", {
      content,
    });
    const agent = this.getClientAgent(this.contextService.context.clientId);
    return await agent.commitToolOutput(content);
  };

  public execute = async (input: string[]) => {
    this.loggerService.logCtx("refundsAgentService execute", {
      input,
    });
    const agent = this.getClientAgent(this.contextService.context.clientId);
    return await agent.execute(input);
  };

  public beginChat = async () => {
    this.loggerService.logCtx("refundsAgentService beginChat");
    const agent = this.getClientAgent(this.contextService.context.clientId);
    return await agent.beginChat();
  };

  public dispose = async () => {
    this.loggerService.logCtx("refundsAgentService dispose");
    const agent = this.getClientAgent(this.contextService.context.clientId);
    await agent.dispose();
    this.getClientAgent.clear(this.contextService.context.clientId);
  };
}

export default RefundsAgentService;
