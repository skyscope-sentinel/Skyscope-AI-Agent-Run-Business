import { ioc } from "src/services";
import { test } from "worker-testbed";

test("Will pass ask_for_agent bug", async (t) => {
    const output = await ioc.specPublicService.complete("Ask for agent function");
    if (await ioc.specPublicService.getAgentName() === "triage-agent") {
        t.pass(`Successfully skipped the navigation output=${output}`);
        return;
    }
    t.fail(`Navigation mistake, agent=${await ioc.specPublicService.getAgentName()} output=${output}`);
});

test("Will navigate to sales agent when user ask the product list", async (t) => {
    const output = await ioc.specPublicService.complete("Give me the product list which you sale");
    if (await ioc.specPublicService.getAgentName() === "sales-agent") {
        t.pass(`Successfully navigated to sales agent output=${output}`);
        return;
    }
    t.fail(`Navigation failed, agent=${await ioc.specPublicService.getAgentName()} output=${output}`);
});

test("Will navigate to refund agent when user ask to withdraw bank deposit", async (t) => {
    const output = await ioc.specPublicService.complete("i got a problem with my previous order. i need to withdraw bank deposit");
    if (await ioc.specPublicService.getAgentName() === "refunds-agent") {
        t.pass(`Successfully navigated to refunds agent output=${output}`);
        return;
    }
    t.fail(`Navigation failed, agent=${await ioc.specPublicService.getAgentName()} output=${output}`);
});

