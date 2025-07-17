import { singleshot } from "functools-kit";
import TYPES from "src/config/types";
import { inject } from "src/core/di";
import RefundsAgentService from "src/services/logic/agent/RefundsAgentService";
import SalesAgentService from "src/services/logic/agent/SalesAgentService";
import TriageAgentService from "src/services/logic/agent/TriageAgentService";

export const getAgentMap = singleshot(() => ({
    "refunds-agent": inject<RefundsAgentService>(TYPES.refundsAgentService),
    "sales-agent": inject<SalesAgentService>(TYPES.salesAgentService),
    "triage-agent": inject<TriageAgentService>(TYPES.triageAgentService),
}));

export const getAgent = (agentName: AgentName) => {
    const agentMap = getAgentMap();
    return agentMap[agentName];
};

export type AgentName = keyof ReturnType<typeof getAgentMap>;

export type Agent = ReturnType<typeof getAgentMap>[AgentName];

export default getAgentMap;
