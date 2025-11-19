import asyncio

if __name__ == "__main__":
    from dotenv import load_dotenv
    assert load_dotenv()

    from docassist.entrypoint import main
    asyncio.run(main())
