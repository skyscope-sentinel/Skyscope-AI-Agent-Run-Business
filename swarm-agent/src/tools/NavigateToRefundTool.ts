import { IAgentTool } from "src/client/ClientAgent";
import { AgentName } from "src/utils/getAgentMap";
import { z } from "zod";
import { ioc } from "src/services";

const PARAMETER_SCHEMA = z.object({}).strict();

export class NavigateToRefundTool implements IAgentTool {

  validate = async (agentName: AgentName, params: Record<string, unknown>) => {
    ioc.loggerService.logCtx("navigateToRefundTool validate", { agentName, params });
    const { success } = await PARAMETER_SCHEMA.spa(params);
    return success;
  };

  call = async (agentName: AgentName) => {
    ioc.loggerService.logCtx("navigateToRefundTool call", { agentName });
    await ioc.connectionPrivateService.commitToolOutput("Navigation success", agentName);
    await ioc.rootSwarmService.setAgent("refunds-agent");
    ioc.connectionPrivateService.emit(
      "Hello. I am a refund agent. Please provide me with the necessary information to process your refund.",
      "refunds-agent"
    );
  };

  getToolSignature = () => ({
    implementation: this.call,
    validate: this.validate,
    type: "function",
    function: {
      name: "navigate_to_refund_tool",
      description: "Navigate to refund agent",
      parameters: {
        type: "object",
        properties: {},
        required: [],
      },
    },
  });
}

export default NavigateToRefundTool;
