from abc import abstractmethod, ABC
from itertools import chain
from os import makedirs, remove, walk
from os.path import join, exists, isdir, abspath, dirname, relpath, basename
from pathlib import Path
from shutil import rmtree
from typing import NamedTuple, Literal, TextIO, Iterable, Self, assert_never


class CodeFilePath(NamedTuple):
    path: str
    """Represent a relative path""" # unless not? it does when used from AnalysedRepo#list_sources() and co
    language: str



EntryType = Literal["file", "directory"]

#todo test it out: https://chatgpt.com/c/68af107c-55f8-8330-9e6d-efaa7fa54062
class ManagedDirectory:
    def __init__(self, root: str | None = None, name: str | None = None):
        self._complete = False
        self.root = root
        self.name = name
        if root is not None and name is not None:
            self._setup()

    def _setup(self):
        assert not self._complete
        assert self.root is not None and self.name is not None
        self.path = join(self.root, self.name)
        if not exists(self.path):
            makedirs(self.path)
        else:
            assert isdir(self.path)
        self._complete = True

    def __str__(self) -> str:
        return type(self).__name__+ "(" + (
            self.absolute_path()
            if self._complete else
            f"root: {self.root}, name: {self.name}"
        ) + ")"

    def __repr__(self):
        return f"{type(self).__name__}(root: {self.root}, name: {self.name}"+("" if self._complete else ", _complete=False")+")"

    def absolute_path(self) -> str:
        return abspath(self.path)

    def subpath(self, *path: str) -> str:
        #only assert it here, the other methods will need to resolve subpath anyway
        assert self._complete
        return join(self.path, *path)

    def exists(self, *path) -> bool:
        return exists(self.subpath(*path))

    def open(self, *path: str, mode: str="r") -> TextIO:
        subp = self.subpath(*path)
        if "w" in mode and not self.exists(subp):
            parent = dirname(subp)
            if not exists(parent):
                makedirs(parent)
        return open(subp, mode)

    def rm(self, *path:  str):
        subpath = self.subpath(*path)
        type_ = self.type_of(*path)
        match type_:
            case "file": remove(subpath)
            case "directory": rmtree(subpath)
            case None: pass
            case _: assert_never(type_)

    def ls(self, *path: str, entry_type: EntryType | Literal["both"] = "both") -> list[str]:
        p = Path(self.subpath(*path))
        if not p.is_dir():
            raise ValueError(f"{path} is not a directory")

        entries = []
        for entry in p.iterdir():
            if entry_type == "file" and not entry.is_file():
                continue
            if entry_type == "directory" and not entry.is_dir():
                continue
            entries.append(entry.name)
        return entries

    # courtesy of ChatGPT
    def tree(self, *path, entry_type: EntryType | Literal["both"] = "both") -> Iterable[str]:
        base = self.subpath(*path)
        root_rel = join(*path) if path else ""  # relative path to starting point

        for dirpath, dirnames, filenames in walk(base):
            # compute relative path with respect to self.path
            rel_dir = relpath(dirpath, self.path)
            if rel_dir == ".":
                rel_dir = ""

            # include directories
            if entry_type in ("directory", "both"):
                for d in dirnames:
                    yield join(rel_dir, d) if rel_dir else d

            # include files
            if entry_type in ("file", "both"):
                for f in filenames:
                    yield join(rel_dir, f) if rel_dir else f

    def type_of(self, *path: str) -> EntryType | None:
        if not self.exists(*path):
            return None
        if isdir(self.subpath(*path)):
            return "directory"
        return "file"

    @classmethod
    def of(cls, *path: str) -> Self:
        joined = join(*path)
        return cls(dirname(joined), basename(joined))

    def subdirectory(self, *path: str) -> Self:
        type_ = self.type_of(*path)
        assert type_ is None or type_ == "directory"
        return ManagedDirectory.of(self.subpath(*path))

RepoItemType = Literal["code", "test", "documentation", "config"] # todo: prompt allows for "test config" and "code and test" too

class AnalysedRepo(ManagedDirectory, ABC):
    def __init__(self, clone_path: str):
        self.clone_path = clone_path
        ManagedDirectory.__init__(self, dirname(clone_path), basename(clone_path))

    #todo add support for examples; these should probably be grouped (multiple files per example)
    #todo add support for build configs (pipeline definitions, scripts that build code or docs, etc)

    @abstractmethod
    def list_sources(self) -> Iterable[CodeFilePath]: ...

    @abstractmethod
    def list_tests(self) -> Iterable[CodeFilePath]: ...

    @abstractmethod
    def list_docs(self) -> Iterable[CodeFilePath]: ...

    @abstractmethod
    def list_configs(self) -> Iterable[CodeFilePath]: ...

    def list(self, item_type, *item_types: RepoItemType) -> Iterable[tuple[RepoItemType, CodeFilePath]]:
        for i_t in set(chain([item_type], item_types)):
            source: Iterable[CodeFilePath] = None
            match i_t:
                case "code": source = self.list_sources()
                case "test": source = self.list_tests()
                case "documentation": source = self.list_docs()
                case "config": source = self.list_configs()
                case _ as never: assert_never(never)
                #todo add examples
            yield from ( (i_t, x) for x in source )

    def list_all(self) -> Iterable[tuple[RepoItemType, CodeFilePath]]:
        return self.list(*RepoItemType.__args__)

class PipOutdatedRepo(AnalysedRepo):
    """
    https://github.com/eight04/pip-outdated
    As checked out at master with HEAD = ac80fcc
    """
    def __init__(self):
        AnalysedRepo.__init__(self, "/home/filip/repos/pip-outdated")

    def list_sources(self) -> Iterable[CodeFilePath]:
        for name in self.tree("pip_outdated"):
            if name.endswith(".py"):
                yield CodeFilePath(name, "python")

    def list_tests(self) -> Iterable[CodeFilePath]:
        for name in self.tree("tests"):
            if name.endswith(".py"):
                yield CodeFilePath(name, "python")

    def list_docs(self) -> Iterable[CodeFilePath]:
        yield CodeFilePath("README.rst", "ReStructuredText")

    def list_configs(self) -> Iterable[CodeFilePath]:
        yield CodeFilePath("setup.py", "python")
        yield CodeFilePath("setup.cfg", "config")

class MitmProxyRepo(AnalysedRepo):
    """
    https://github.com/mitmproxy/mitmproxy
    As checked out at master with HEAD = 37d1cf5
    """
    def __init__(self):
        AnalysedRepo.__init__(self, "/home/filip/repos/mitmproxy")

    def list_sources(self) -> Iterable[CodeFilePath]:
        for name in self.tree("mitmproxy"):
            if name.endswith(".py"):
                yield CodeFilePath(name, "python")

    def list_tests(self) -> Iterable[CodeFilePath]:
        for name in self.tree("test"):
            if name.endswith(".py"):
                yield CodeFilePath(name, "python")

    def list_docs(self) -> Iterable[CodeFilePath]:
        yield CodeFilePath("README.md", "markdown")
        yield CodeFilePath("SECURITY.md", "markdown")
        yield CodeFilePath("CHANGELOG.md", "markdown")
        for name in self.tree("docs"):
            if name.endswith(".md"):
                yield CodeFilePath(name, "markdown")

    def list_configs(self) -> Iterable[CodeFilePath]:
        yield CodeFilePath("pyproject.toml", "toml")


class FastApiUserRepo(AnalysedRepo):
    """
    https://github.com/fastapi-users/fastapi-users
    As checked out at master with HEAD = 9d78b2a
    """
    def __init__(self):
        AnalysedRepo.__init__(self, "/home/filip/repos/fastapi-users")

    def list_sources(self) -> Iterable[CodeFilePath]:
        for name in self.tree("fastapi_users"):
            if name.endswith(".py"):
                yield CodeFilePath(name, "python")

    def list_tests(self) -> Iterable[CodeFilePath]:
        for name in self.tree("tests"):
            if name.endswith(".py"):
                yield CodeFilePath(name, "python")

    def list_docs(self) -> Iterable[CodeFilePath]:
        yield CodeFilePath("README.md", "markdown")
        yield CodeFilePath("mkdocs.yml", "yaml")
        for name in self.tree("docs"):
            if name.endswith(".md"):
                yield CodeFilePath(name, "markdown")
            elif name.endswith(".py"):
                yield CodeFilePath(name, "python")

    def list_configs(self) -> Iterable[CodeFilePath]:
        yield CodeFilePath("pyproject.toml", "toml")
        yield CodeFilePath("setup.cfg", "cfg")


class SpringPetclinicRepo(AnalysedRepo):
    """
    https://github.com/spring-projects/spring-petclinic
    As checked out at main with HEAD = 6feeae0
    """
    def __init__(self):
        AnalysedRepo.__init__(self, "/home/filip/repos/spring-petclinic")

    def list_sources(self) -> Iterable[CodeFilePath]:
        for name in self.tree("src/main"):
            if name.endswith(".java"):
                yield CodeFilePath(name, "java")
            if name.endswith(".properties"):
                yield CodeFilePath(name, "properties")
            if name.endswith(".txt"):
                yield CodeFilePath(name, "plaintext")

    def list_tests(self) -> Iterable[CodeFilePath]:
        for name in self.tree("src/test"):
            if name.endswith(".java"):
                yield CodeFilePath(name, "java")
            if name.endswith(".properties"):
                yield CodeFilePath(name, "properties")
            if name.endswith(".jmx"):
                yield CodeFilePath(name, "jmeter")


    def list_docs(self) -> Iterable[CodeFilePath]:
        yield CodeFilePath("README.md", "markdown")

    def list_configs(self) -> Iterable[CodeFilePath]:
        yield CodeFilePath("build.gradle", "gradle")
        yield CodeFilePath("settings.gradle", "gradle")
        yield CodeFilePath("pom.xml", "maven")
        yield CodeFilePath("docker-compose.yml", "docker-compose")
        for name in self.tree("src/checkstyle"):
            if name.endswith(".xml"):
                yield CodeFilePath(name, "checkstyle")
        for name in self.tree("k8s"):
            if name.endswith(".yml"):
                yield CodeFilePath(name, "kubernetes")
