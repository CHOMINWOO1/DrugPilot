# Credentials and private configuration

Never commit API keys, access tokens, passwords, private keys, local account details, private server addresses, service-account files, or database/session exports.

Use environment variables or an ignored `.env` file. Only placeholder values belong in `.env.example`. The committed `.gitignore` excludes local credentials, state, and common secret-bearing files.

`.gitignore` does not remove secrets already committed or embedded in other files. Review the staged diff before each push. If a real credential is exposed, revoke/rotate it and remove it from Git history as well.

This publication starts from a curated source snapshot with no inherited local Git history. Historical workstation usernames, hostnames, and private network addresses were generalized. Explicit dummy credentials in security tests remain fixtures. Loopback addresses and documented example addresses are not deployment endpoints.

The checks performed for this snapshot are recorded in `VALIDATION.md`. No automated scan can guarantee the absence of every form of sensitive information.
