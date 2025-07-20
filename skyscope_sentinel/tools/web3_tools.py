from crewai_tools import BaseTool

class GetBalanceTool(BaseTool):
    name: str = "Get Balance"
    description: str = "Gets the balance of the agent's wallet."

    def _run(self, agent):
        return agent.web3_utils.get_balance()

class SendTransactionTool(BaseTool):
    name: str = "Send Transaction"
    description: str = "Sends a transaction to the specified address."

    def _run(self, agent, to_address: str, amount: float):
        return agent.web3_utils.send_transaction(to_address, amount)

class GetSolanaKeypairTool(BaseTool):
    name: str = "Get Solana Keypair"
    description: str = "Gets the Solana keypair for the agent's wallet."

    def _run(self, agent):
        return agent.web3_utils.get_solana_keypair()
