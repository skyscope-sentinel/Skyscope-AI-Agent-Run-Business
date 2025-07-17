import { IAgentTool } from "src/client/ClientAgent";
import { AgentName } from "src/utils/getAgentMap";
import { z } from "zod";
import { ioc } from "src/services";

const PARAMETER_SCHEMA = z.object({}).strict();

export class NavigateToSalesTool implements IAgentTool {

  validate = async (agentName: AgentName, params: Record<string, unknown>) => {
    ioc.loggerService.logCtx("navigateToSalesTool validate", { agentName, params });
    const { success } = await PARAMETER_SCHEMA.spa(params);
    return success;
  };

  call = async (agentName: AgentName) => {
    ioc.loggerService.logCtx("navigateToSalesTool call", { agentName });
    await ioc.connectionPrivateService.commitToolOutput("Navigation success", agentName);
    await ioc.rootSwarmService.setAgent("sales-agent");
    ioc.connectionPrivateService.emit(
      "Hello. I am a sales agent. Please provide me with the necessary information to process your sale.",
      "sales-agent"
    );
  };

  getToolSignature = () => ({
    validate: this.validate,
    implementation: this.call,
    type: "function",
    function: {
      name: "navigate_to_sales_tool",
      description: "Navigate to sales agent",
      parameters: {
        type: "object",
        properties: {},
        required: [],
      },
    },
  });
}

export default NavigateToSalesTool;
