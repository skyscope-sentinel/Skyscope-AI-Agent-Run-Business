import { memoize } from "functools-kit";
import ClientAgent, { IAgent } from "src/client/ClientAgent";
import TYPES from "src/config/types";
import { inject } from "src/core/di";
import { TContextService } from "src/services/base/ContextService";
import LoggerService from "src/services/base/LoggerService";
import NavigationRegistryService from "src/services/function/NavigationRegistryService";

const AGENT_PROMPT = `You are to triage a users request, and call a tool to transfer to the right agent.
To transfer use a right tool from a list. Use the chat history instead of asking a direct question
Do not tell the user the details of your functionality
Act like a real person
Navigate to the agent without asking additional details
If the speech is about agent, navigate immediately
If you can't be sure which agent you should navigate to, ask the direct question
`;

export class TriageAgentService implements IAgent {
  readonly contextService = inject<TContextService>(TYPES.contextService);
  readonly loggerService = inject<LoggerService>(TYPES.loggerService);

  readonly navigationRegistryService = inject<NavigationRegistryService>(TYPES.navigationRegistryService);

  private getClientAgent = memoize(
    ([clientId]) => clientId,
    (clientId: string) =>
      new ClientAgent({
        clientId,
        agentName: "triage-agent",
        prompt: AGENT_PROMPT,
        tools: [
          this.navigationRegistryService.useNavigateToRefund(),
          this.navigationRegistryService.useNavigateToSales(),
        ],
      })
  );

  public execute = async (input: string[]) => {
    this.loggerService.logCtx("triageAgentService execute", {
      input,
    });
    const agent = this.getClientAgent(this.contextService.context.clientId);
    return await agent.execute(input);
  };

  public beginChat = async () => {
    this.loggerService.logCtx("triageAgentService beginChat");
    const agent = this.getClientAgent(this.contextService.context.clientId);
    return await agent.beginChat();
  };

  commitSystemMessage = async (message: string) => {
    this.loggerService.logCtx("triageAgentService commitSystemMessage", {
      message,
    });
    const agent = this.getClientAgent(this.contextService.context.clientId);
    return await agent.commitSystemMessage(message);
  };

  commitToolOutput = async (content: string) => {
    this.loggerService.logCtx("triageAgentService commitToolOutput", {
      content,
    });
    const agent = this.getClientAgent(this.contextService.context.clientId);
    return await agent.commitToolOutput(content);
  };

  public dispose = async () => {
    this.loggerService.logCtx("triageAgentService dispose");
    const agent = this.getClientAgent(this.contextService.context.clientId);
    await agent.dispose();
    this.getClientAgent.clear(this.contextService.context.clientId);
  };
}

export default TriageAgentService;
