import TYPES from "../config/types";
import "../config/provide";
import { inject, init } from "../core/di";
import type LoggerService from "./base/LoggerService";
import ErrorService from "./base/ErrorService";
import { TContextService } from "./base/ContextService";
import RefundsAgentService from "./logic/agent/RefundsAgentService";
import SalesAgentService from "./logic/agent/SalesAgentService";
import TriageAgentService from "./logic/agent/TriageAgentService";
import RootSwarmService from "./logic/RootSwarmService";
import ConnectionPublicService from "./public/ConnectionPublicService";
import ConnectionPrivateService from "./private/ConnectionPrivateService";
import EmbeddingService from "./api/EmbeddingService";
import CompletionService from "./api/CompletionService";
import MongooseService from "./base/MongooseService";
import RedisService from "./base/RedisService";
import ClientSwarmDbService from "./db/ClientSwarmDbService";
import ProductDbService from "./db/ProductDbService";
import MigrationPublicService from "./public/MigrationPublicService";
import MigrationPrivateService from "./private/MigrationPrivateService";
import ClientHistoryDbService from "./db/ClientHistoryDbService";
import ClientCartDbService from "./db/ClientCartDbService";
import SpecPrivateService from "./private/SpecPrivateService";
import SpecPublicService from "./public/SpecPublicService";

const apiServices = {
    embeddingService: inject<EmbeddingService>(TYPES.embeddingService),
    completionService: inject<CompletionService>(TYPES.completionService),
};

const baseServices = {
    loggerService: inject<LoggerService>(TYPES.loggerService),
    errorService: inject<ErrorService>(TYPES.errorService),
    contextService: inject<TContextService>(TYPES.contextService),
    mongooseService: inject<MongooseService>(TYPES.mongooseService),
    redisService: inject<RedisService>(TYPES.redisService),
};

const publicServices = {
    connectionPublicService: inject<ConnectionPublicService>(TYPES.connectionPublicService),
    migrationPublicService: inject<MigrationPublicService>(TYPES.migrationPublicService),
    specPublicService: inject<SpecPublicService>(TYPES.specPublicService),
};

const privateServices = {
    connectionPrivateService: inject<ConnectionPrivateService>(TYPES.connectionPrivateService),
    migrationPrivateService: inject<MigrationPrivateService>(TYPES.migrationPrivateService),
    specPrivateService: inject<SpecPrivateService>(TYPES.specPrivateService),
};

const dbServices = {
    clientSwarmDbService: inject<ClientSwarmDbService>(TYPES.clientSwarmDbService),
    clientHistoryDbService: inject<ClientHistoryDbService>(TYPES.clientHistoryDbService),
    clientCartDbService: inject<ClientCartDbService>(TYPES.clientCartDbService),
    productDbService: inject<ProductDbService>(TYPES.productDbService),
};

const logicServices = {
    refundsAgentService: inject<RefundsAgentService>(TYPES.refundsAgentService),
    salesAgentService: inject<SalesAgentService>(TYPES.salesAgentService),
    triageAgentService: inject<TriageAgentService>(TYPES.triageAgentService),
    rootSwarmService: inject<RootSwarmService>(TYPES.rootSwarmService),
};

init();

export const ioc = {
    ...apiServices,
    ...baseServices,
    ...publicServices,
    ...privateServices,
    ...dbServices,
    ...logicServices,
};

Object.assign(globalThis, { ioc });
