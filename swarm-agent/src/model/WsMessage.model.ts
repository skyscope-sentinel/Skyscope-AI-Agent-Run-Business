import { AgentName } from "src/utils/getAgentMap";

export interface IIncomingMessage {
  clientId: string;
  stamp: string;
  data: string[];
  agentName: AgentName;
}

export interface IOutgoingMessage {
  clientId: string;
  stamp: string;
  data: string;
  agentName: AgentName;
}
