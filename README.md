# doc-agent

> Also known as "docassist". That wa the original name, still present in code, but I decided to
> rewrite from scratch in empty repo, as the old one carried a lot of big files.

__This is a learning project. It is not well-written, clean code was not the priority. It's main
purpose was not a product, but rather an opportunity to tackle things that I expected to be 
valuable. If it ever were to become a product, it would need a holistic cleanup, if not a rewrite.
As such, my phrasing may not be work appropriate - in the end I wrote it to myself and never 
expected this to be presented. Pardon vulgarisms, I use them to either emphasize something or
to relieve frustration.__

The idea here was that the app takes a project and generates Spring-quality (as in Spring in Java
world) documentation out of it. I don't mean Sphinx-like docs, I dont need code reference - you 
could embed such things in the outcome, but I mean the docs with proper introduction, licensing and
versioning, installation, code reference I already mentioned above, but also examples of usage, 
some sort of FAQ, etc. Those are the parts that make great OSS products (Spring, Python `requests`,
`fastapi`, `pydantic`,...) - its not just code, its also the semantic layer over that code.

In professional setup you don't always have the time to prepare this. I wanted to hand this off 
to language models, so that developers could focus on code and just review the changes to the
docs together with code changes, where docs would be generated/updated docs would happen in CI.

This was my first project with code-reading agents so I wanted to build knowledge layer myself.
Now I would probably use [code-review-graph](https://github.com/tirth8205/code-review-graph)
or something like that.

The implementation was supposed to take something like [spec.py](docassist/structure/spec.py) 
as an input, along with the path to repository (see [subject.py](docassist/subjects.py)) and 
resolve the spec against the project with minimal metadata (what is code, what is test code, 
what are the available configs, etc). 

I've built a graph of very dedicated and limited agents to read over the codebase, take notes
from different perspectives (developer/non-developer x maintainer/consumer), chunk them (with
custom markdown chunking, built on assumption of short paragraphs, with redundancy based on
chapter path of the chunk) and build an index out of them. I put a lot of effort into building
a consistent metadata layer over the chunks, already expecting that I'll need them for 
disambiguation and CDC.

CDC was not implemented yet, but to avoid non-repeatable runs I built a model sampling layer.
I say "sampling" and not "caching", as I believe caching to be appropriate for 
deterministically changing data, while LLMs are effectively random variables parametrized with
their input, so this is basically sampling of the variable and storing the samples for reuse. 

I hope to circle back to the sampling layer and extract it to an external project at some point.
For now it only captures the text responses from the model (so, no tool calls, no thinking) and
doesn't support streaming, but it does provide fundamental abstractions for sampler and proves
that it's doable with pydantic-ai.

> The sampling layer was important to me, as it deferred actual CDC a lot. Before I picked pydantic-ai
> as framework of choice, I reviewed a bunch of other frameworks against the ease of implementing
> such sampling.

On top of this I've built a retrieval pipeline that I exposed to some agents as a tool. Unlike
the raw RAG idea, it included the reason for search along with the query (its different when 
you "search for things like 'foo'" than when you "search for things like "foo", so that I can see 
if they bar"). The pipeline included search surface explosion (query rephrasing, I guess), agentic
deduplication (agent was picking the best duplicate to keep given the search reason) and reranking.

At some point I started getting lost in natural-language prompts that shared common parts (like 
perspective from which the notes are taken). Because of that I started looking into structured 
prompting (not as a framework to gather effective prompting practices, but rather as practice
of phrasing the prompt in JSON-like structure that gets dumped to a common format). That resulted
in [`lmxml`](https://github.com/FilipMalczak/lmxml) project getting extracted and published. I 
decided on XML, as (a) it reads well for LLMs and (b) it doesn't mix with the content I embed in 
prompts (because the files I generate or read are rarely XML; embedding YAML in YAML prompt
can lead to weird and misleading formatting, while YAML inside XML delimiters has much lower 
cognitive overhead). The original code is still in this repo as simple_xml.py, as I extracted it
when I was starting another project.

I didn't invest time in any automated evals (that is a team-scoped work, for learning project
it would just be an overkill), but the resolver (built from heavily specialized agents, each 
handling different kind of modal structure in the spec, like "resolve variable", "decide on closed
question", "answer an open question", "generate specific content") was pretty effective, on
less-demanding parts. 

I gave up on this project, as I wanted to move on to more agentic and less specialized LLM 
applications. Besides looking for greener grass, I got a little disheartened, because while my system
was able to resolve variables like `let(project_name="current project name")` or pick an answer 
for `depends_on_question` clauses (see spec.py), it failed badly on `expand_on`. The reason for that
failure was compound - I used too small models, I resolved each specification node in isolation
(so when resolving `expand_on` I didn't know how I got to previous resolves, like variables), 
I didn't write the spec well enough, ditto prompts (to list only a few factors).

## TL;DR

### What I learned

- pydantic-ai and pydantic-graph
- how to work with LLMs in general (as mentioned, this was my first personal project in this domain)
- why async and streaming is the way to go with agents
- that structured prompting doesn't hurt LLM efficiency (even for small models), while enhancing DX
- how to use logfire and how to read it
- good RAG practices and what challenges to expect down the RAG road

> I already had experience with LanceDB and Chroma from the past, I don't consider FAISS as lesson learned here. 

### What worked

- RAG pipeline
  - custom markdown chunking
  - rephrase/deduplicate/rerank flow with search purpose as additional argument
- simple sync model sampling
- [`lmxml`](https://github.com/FilipMalczak/lmxml) utility for stuctured prompts

### What didn't work

- knowing what I know, I would go with different architecture, that allows agents to explore more liberally, instead
  of handcrafting the behaviour graphs
- models I picked were probably too small and not capable enough to come up with high-level concepts (like project use cases)
