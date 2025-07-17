export interface CompleteRequest {
    clientId: string;
    serviceName: string;
    requestId: string;
    messages: string[];
}

export default CompleteRequest;
