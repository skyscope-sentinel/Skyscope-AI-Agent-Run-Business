import { memoize } from "functools-kit";
import { TContextService } from "../base/ContextService";
import { inject } from "src/core/di";
import TYPES from "src/config/types";
import LoggerService from "../base/LoggerService";
import ClientCart, { ICart } from "src/client/ClientCart";
import { AgentName } from "src/utils/getAgentMap";
import { ICartItem } from "src/model/CartItem.model";

type TCart = {
  [key in keyof ICart]: any;
};

export class ClientCartDbService implements TCart {
  readonly loggerService = inject<LoggerService>(TYPES.loggerService);
  readonly contextService = inject<TContextService>(TYPES.contextService);

  private getClientCart = memoize(
    ([clientId, agentName]) => `${clientId}-${agentName}`,
    (clientId: string, agentName: AgentName) =>
      new ClientCart({
        clientId,
        agentName,
      })
  );

  public length = async (agentName: AgentName) => {
    this.loggerService.logCtx("clientCartDbService length");
    return await this.getClientCart(
      this.contextService.context.clientId,
      agentName
    ).length();
  };

  public toArray = async (agentName: AgentName) => {
    this.loggerService.logCtx("clientCartDbService toArrayForAgent");
    return await this.getClientCart(
      this.contextService.context.clientId,
      agentName
    ).toArray();
  };

  public push = async (agentName: AgentName, message: ICartItem) => {
    this.loggerService.logCtx("clientCartDbService push", { message });
    return await this.getClientCart(
      this.contextService.context.clientId,
      agentName
    ).push(message);
  };

  public pop = async (agentName: AgentName) => {
    this.loggerService.logCtx("clientCartDbService pop");
    return await this.getClientCart(
      this.contextService.context.clientId,
      agentName
    ).pop();
  };

  public clear = async (agentName: AgentName) => {
    this.loggerService.logCtx("clientCartDbService clear");
    return await this.getClientCart(
      this.contextService.context.clientId,
      agentName
    ).clear();
  };

  public dispose = async (agentName: AgentName) => {
    this.loggerService.logCtx("clientCartDbService dispose");
    await this.getClientCart(
      this.contextService.context.clientId,
      agentName
    ).dispose();
    this.getClientCart.clear(this.contextService.context.clientId);
  };
}

export default ClientCartDbService;
