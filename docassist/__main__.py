if __name__ == "__main__":
    from dotenv import load_dotenv
    assert load_dotenv()

    import asyncio

    import logfire

    # def scrubbing_callback(m: logfire.ScrubMatch):
    #     should_allow = False
    #     should_allow |=  m.path[:2] == ("attributes", "input_data")
    #     should_allow |=  m.path[:2] == ("attributes", "result")
    #     should_allow |=  m.path[-1] == "content"
    #     should_allow |=  m.pattern_match.group(0)
    #     if should_allow:
    #         return m.value
    #
    #
    # logfire.configure(service_name="docassist", scrubbing=ScrubbingOptions(callback=scrubbing_callback))
    logfire.configure(
        service_name="docassist",
        scrubbing=False, # do not scrub, or you'll lose anything session-related or anything that mentions auth - content of notes included
        environment="local",
        # send_to_logfire=False, # this is here if you needed to debug OTEL again; otherwise, keep commented out
    )
    # logfire.instrument_pydantic() # this doesn't help; it actually makes the spans less readable
    logfire.instrument_pydantic_ai()
    #fixme this adds spans over actual openai calls, but embedder is not pydantic-ai abstraction, so its half-assed
    #fixme chat completion cost is not easily visible neither
    logfire.instrument_openai()
    logfire.instrument_httpx(
        capture_headers=False, # since we don't scrub, we shouldn't expose headers that might contain API keys
    )
    logfire.install_auto_tracing(
        "docassists[.].*",
        min_duration=0.01
    )
    logfire.instrument_print()
    logfire.instrument_system_metrics()

    from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor

    AsyncioInstrumentor().instrument()

    from pydantic_ai.agent import Agent

    Agent.instrument_all()

    from docassist.entrypoint import main

    with logfire.span("docassist"):
        asyncio.run(main())
