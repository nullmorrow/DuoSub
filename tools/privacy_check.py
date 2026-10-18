"""Fail closed on private files, personal identity, credentials, or wrong Git identity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_NAME = 'nullmorrow'
PUBLIC_EMAIL = '344519723+nullmorrow@users.noreply.github.com'
EMAIL = re.compile(r'(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b')
HOME = re.compile(r'(?i)(?:[a-z]:[/\\]+Users[/\\]+(?!Public\b|Default\b)[\w.-]+|/(?:home|Users)/[\w.-]+)')
SECRET = re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----)')
PRIVATE = re.compile(r'(?i)^(?:Projects|history|migration|baseline|reports)(?:/|$)|(?:^|/)(?:cache|__pycache__|journal|exports|\.git)(?:/|$)|(?:^|/)(?:AGENTS\.md|WORKLOG\.md|HANDOFF[^/]*|MIGRATION[^/]*|.*\.local\.json|\.env(?:\..*)?|.*\.(?:sqlite3?|db|log|wav|mkv|mp4|dll|exe|zip|bundle|lnk))$')
INTERNAL = re.compile(r'(?i)^(?:tools/eval/data/|docs/eval/|docs/(?:CLAUDE-HANDOFF|CODEX-REVIEW|CHECKPOINT|GPT-|SPEC-|RUBY-|WEEK_FIXES)|docs/ui-redesign/(?:current/|PROGRESS))')


def blocked_values() -> list[str]:
    """A private legacy-identity list may be retained locally, never committed."""
    path = ROOT / '.privacy-blocklist.local.json'
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return []
    result = data.get('blocked', [])
    commits = data.get('old_commits', [])
    return [value.casefold() for value in result + commits if isinstance(value, str) and len(value) >= 4]


def content_issues(name: str, data: bytes, blocked=()) -> list[str]:
    issues = []
    if PRIVATE.search(name) or INTERNAL.search(name):
        issues.append('private or generated file')
    # Binary assets are limited to icons; no arbitrary documents/archives can bypass scanning.
    try:
        text = data.decode('utf-8-sig')
    except UnicodeDecodeError:
        if not name.startswith('assets/') or Path(name).suffix not in ('.ico', '.png'):
            issues.append('unreviewed binary')
        text = data.decode('latin-1')
    if any(match.group() != PUBLIC_EMAIL for match in EMAIL.finditer(text)):
        issues.append('email address')
    if HOME.search(text):
        issues.append('personal home path')
    if SECRET.search(text):
        issues.append('credential or private key')
    folded = text.casefold()
    if any(value in folded or value in name.casefold() for value in blocked):
        issues.append('legacy private identifier')
    return issues


def git(*args, text=False):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=text, encoding='utf-8' if text else None)


def check_identity(value: str) -> bool:
    return value.startswith(f'{PUBLIC_NAME} <{PUBLIC_EMAIL}> ')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cached', action='store_true')
    parser.add_argument('--history', action='store_true')
    parser.add_argument('--message', type=Path)
    parser.add_argument('--remote')
    args = parser.parse_args()
    errors = []
    blocked = blocked_values()
    if args.remote and not re.fullmatch(r'(?:https://github\.com/|git@github\.com:)nullmorrow/DuoSub(?:\.git)?', args.remote):
        errors.append(('remote', ['unauthorized push destination']))
    if args.message:
        issues = content_issues('commit-message.txt', args.message.read_bytes(), blocked)
        if issues:
            errors.append(('commit message', issues))
    if args.cached:
        for name in git('ls-files', '-z').split(b'\0'):
            if not name:
                continue
            rel = name.decode('utf-8')
            issues = content_issues(rel, git('show', ':' + rel), blocked)
            if issues:
                errors.append((rel, issues))
        for field in ('GIT_AUTHOR_IDENT', 'GIT_COMMITTER_IDENT'):
            if not check_identity(git('var', field, text=True).strip()):
                errors.append((field, ['incorrect public identity']))
    if args.history:
        seen = set()
        for commit in git('rev-list', '--all', text=True).splitlines():
            for name, value in zip(('author', 'committer'), git('show', '-s', '--format=%an <%ae>%n%cn <%ce>', commit, text=True).splitlines()):
                if value != f'{PUBLIC_NAME} <{PUBLIC_EMAIL}>':
                    errors.append((commit[:12] + ' ' + name, ['incorrect historical identity']))
            message = git('show', '-s', '--format=%B', commit)
            issues = content_issues('commit-message.txt', message, blocked)
            if issues:
                errors.append((commit[:12] + ' message', issues))
            for record in git('ls-tree', '-rz', commit).split(b'\0'):
                if record:
                    meta, name = record.split(b'\t', 1)
                    rel = name.decode('utf-8')
                    blob = meta.split()[2].decode('ascii')
                    issues = content_issues(rel, b'', ())
                    if (blob, rel) not in seen:
                        seen.add((blob, rel))
                        issues.extend(content_issues(rel, git('cat-file', 'blob', blob), blocked))
                    if issues:
                        errors.append((commit[:12] + ':' + rel, issues))
        for record in git('for-each-ref', '--format=%(objecttype) %(objectname) %(refname)', 'refs/tags', text=True).splitlines():
            kind, oid, ref = record.split(' ', 2)
            issues = content_issues('tag-reference.txt', ref.encode('utf-8'), blocked)
            if kind == 'tag':
                payload = git('cat-file', 'tag', oid)
                issues.extend(content_issues('tag-message.txt', payload, blocked))
                tagger = re.search(rb'^tagger (.+)$', payload, re.M)
                if not tagger or not check_identity(tagger.group(1).decode('utf-8')):
                    issues.append('incorrect tagger identity')
            if issues:
                errors.append((ref, issues))
    for name, issues in errors:
        print(name + ': ' + ', '.join(issues))
    if errors:
        print(f'Privacy check failed: {len(errors)} entries. Values are intentionally not printed.')
        return 1
    print('Privacy check passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
