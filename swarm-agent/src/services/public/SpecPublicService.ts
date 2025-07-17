import { inject } from "src/core/di";
import SpecPrivateService from "../private/SpecPrivateService";
import LoggerService from "../base/LoggerService";
import TYPES from "src/config/types";
import ContextService, { IContext } from "../base/ContextService";
import { randomString } from "functools-kit";
import { AgentName } from "src/utils/getAgentMap";

interface ISpecPrivateService extends SpecPrivateService {}

type TSpecPrivateService = {
  [key in keyof ISpecPrivateService]: any;
};

const TESTBED_CONTEXT: IContext = {
  clientId: `testbed-${randomString()}`,
};

export class SpecPublicService implements TSpecPrivateService {
  private readonly loggerService = inject<LoggerService>(TYPES.loggerService);

  private readonly specPrivateService = inject<SpecPrivateService>(
    TYPES.specPrivateService
  );

  public complete = async (msg: string) => {
    this.loggerService.log("specPublicService complete", { msg });
    return await ContextService.runInContext(async () => {
      return await this.specPrivateService.complete(msg);
    }, TESTBED_CONTEXT);
  };

  public getAgentName = async () => {
    this.loggerService.log("specPublicService getAgentName");
    return await ContextService.runInContext(async () => {
      return await this.specPrivateService.getAgentName();
    }, TESTBED_CONTEXT);
  };

  public getAgent = async () => {
    this.loggerService.log("specPublicService getAgent");
    return await ContextService.runInContext(async () => {
      return await this.specPrivateService.getAgent();
    }, TESTBED_CONTEXT);
  };

  public setAgent = async (agentName: AgentName) => {
    this.loggerService.log("specPublicService setAgent", { agentName });
    return await ContextService.runInContext(async () => {
      return await this.specPrivateService.setAgent(agentName);
    }, TESTBED_CONTEXT);
  };

  public compareStrings = async (a: string, b: string) => {
    this.loggerService.logCtx("specPrivateService compareStrings", { a, b });
    return await ContextService.runInContext(async () => {
      return await this.specPrivateService.compareStrings(a, b);
    }, TESTBED_CONTEXT);
  };
}

export default SpecPublicService;
