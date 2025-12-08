from pprint import pformat

from docassist.structure.model import let, chapter, article, depends_on_question, expand_on

logger = get_logger(__name__)

#todo this should be enhanced
_interface_article_description = """
Explanations on usage of the exposed interface. Should be complete, but focused on usage details. Short examples
are welcome for the sake of explanations, but the longer one will be presented in another, dedicated chapter.
Should be focused on single exposed interface. May refer to other interfaces to clarify some use case or note that
something can be done in multiple ways. Usage should be described both in the terms of how to achieve
something, as well as how will this project behave in such case.
"""


root_specification = let(
    project_name="current project name"
) @ chapter(
        "{project_name}",
        preamble_description="Short summary of this project, its intended audience and key use cases",
        afterword_description="License and author footer",
        content=[
            article(
                "Introduction",
                """
                Structured introduction into why and by whom this project may be used, with necessary explanations,
                refreshers and anecdotes. Should give the reader general gist of what he or she might find in this 
                documentation.
                """
            ),
            article(
                "Installation",
                """
                Description of how to get a working copy of this project. This is the place to explain which package index
                (like Maven Central or PyPI) to target and what are the names or coordinates of artifacts exposed
                by this project. Artifacts should include (but not limited to) JAR/WAR/EAR files, wheel/egg files, gemfiles,
                OCI images or other container images, as well as any namespaces (like XMLNS or JNDI). 
                """
            ),
            depends_on_question(
                "Is there one or more interfaces exposed by this project? Non-exhaustive list of examples of interfaces is: "
                "CLI, HTTP API, RPC API.",
                {
                    "there are no interfaces exposed by this project": None,
                    "there is exactly one interface exposed by this project": let(
                            interface_name="the name of exposed interface, as suitable for headers and titles"
                        ) @ article(
                            "Usage - {interface_name}",
                            _interface_article_description
                        ),
                    "there is more than one interface exposed by this project": chapter(
                        "Usage",
                        preamble_description="short commentary on the number and names/types of exposed interfaces as well as their intended usage; high-level perspective",
                        content=[
                            expand_on("all the interfaces exposed by this project;  non-exhaustive list of examples of "
                                      "interfaces is: CLI, HTTP API, RPC API.") & let(
                                interface_name="the name of exposed interface, as suitable for headers and titles"
                            ) @ article(
                                "{interface_name}",
                                _interface_article_description
                            )
                        ]
                    )
                }
            ),
            chapter(
                "Examples",
                content=[
                    expand_on("at least 3 and at most 10 of the most important use cases for this project; produce"
                              "as many examples as you can within that limit and without repeating yourself") & let(
                        example_slug="example name in lower_snake_case format",
                        example_name="human-readable name of the example, as suitable for headers and titles",
                        example_instructions="the body of generated use case/scenario/example, as numbered, possibly "
                                         "multi-level list of steps required to perform that scenario; may include "
                                         "prerequisites; must represent a scenario described by example_slug and "
                                         "example_name"
                    ) @ article(
                        "{example_name}",
                        "Description of the example, where the example scenario is:\n{example_instructions}\n\nShould be "
                        "exhaustive and step-down (starting from the general idea, going down towards more granulated"
                        "concepts)."
                    )
                ]
            )
        ]
    )

logger.info("Root specification:")
logger.info("\n"+pformat(root_specification, width=120))