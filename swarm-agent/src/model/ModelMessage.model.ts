import { AgentName } from "src/utils/getAgentMap";

export interface IModelMessage {
    role: 'assistant' | 'system' | 'tool' | 'user';
    agentName: AgentName;
    content: string;
    tool_calls?: {
        function: {
            name: string;
            arguments: {
                [key: string]: any;
            };
        };
    }[];
}
