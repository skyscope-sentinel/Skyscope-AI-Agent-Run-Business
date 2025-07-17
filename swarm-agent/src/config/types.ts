const apiServices = {
    completionService: Symbol('completionService'),
    embeddingService: Symbol('embeddingService'),
};

const baseServices = {
    loggerService: Symbol('loggerService'),
    errorService: Symbol('errorService'),
    contextService: Symbol('contextService'),
    mongooseService: Symbol('mongooseService'),
    redisService: Symbol('redisService'),
};

const publicServices = {
    connectionPublicService: Symbol('connectionPublicService'),
    migrationPublicService: Symbol('migrationPublicService'),
    specPublicService: Symbol('specPublicService'),
};

const privateServices = {
    connectionPrivateService: Symbol('connectionPrivateService'),
    migrationPrivateService: Symbol('migrationPrivateService'),
    specPrivateService: Symbol('specPrivateService'),
};

const dbServices = {
    clientSwarmDbService: Symbol('clientSwarmDbService'),
    clientHistoryDbService: Symbol('clientHistoryDbService'),
    clientCartDbService: Symbol('clientCartDbService'),
    productDbService: Symbol('productDbService'),
};

const functionServices = {
    navigationRegistryService: Symbol('navigationRegistryService'),
};

const logicServices = {
    refundsAgentService: Symbol('refundsAgentService'),
    salesAgentService: Symbol('salesAgentService'),
    triageAgentService: Symbol('triageAgentService'),
    rootSwarmService: Symbol('rootSwarmService'),
};

export const TYPES = {
    ...apiServices,
    ...baseServices,
    ...publicServices,
    ...privateServices,
    ...dbServices,
    ...functionServices,
    ...logicServices,
};

export default TYPES;
