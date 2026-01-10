from docassist.index.document import Document
from docassist.parametrized import Parametrized
from docassist.preindexing.perspectives import perspective, PERSPECTIVES
from docassist.structured_agent import WriterAgent
from docassist.system_prompts import PromptingTask


file_note_taker = Parametrized(
    PERSPECTIVES,
    lambda name_suffix, params:
        WriterAgent(
            name="note taker that handles single file"+name_suffix,
            persona="note taker",
            perspective=perspective(**params),
            task=PromptingTask(
                high_level="take notes from the input file",
                low_level="take notes from given perspective or reply with 'N/A' to indicate no meaningful content from "
                          "that perspective; you should not look into the dependent and depending files, your sole focus "
                          "should be on currently processed file",
                detailed="take notes that can be later used to prepare user-facing documentation of the project "
                         "that the input file is part of; at this stage you should take the notes from a single perspective "
                         "(this will happen for any possible perspective in parallel); do not note things that are not "
                         "applicable to the current perspective; if there are no notes to be taken from given perspective,"
                         "use the `empty_result` tool to indicate and explain that",
                context="you're reading the whole project for the first time; you need to extract the useful information for later usage"
            ),
            input_type=Document,
            allow_no_result=True,
            output_format="Markdown"
        )
)