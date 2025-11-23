from logfire import span
from namesgenerator import get_random_name

from docassist.config import CONFIG
from docassist.processing_graph import process_directory


async def main():
    session_name = get_random_name()
    print("Session name:", session_name)
    with span("session") as s:
        s.set_attribute("session.name", session_name)
        for repo in CONFIG.repos:
            await process_directory.run(inputs=repo)