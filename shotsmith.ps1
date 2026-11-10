# Shotsmith terminal application. Existing AVS CLI commands are forwarded unchanged.
[CmdletBinding()]
param([Parameter(ValueFromRemainingArguments=$true)][string[]]$CommandArgs)
if ($null -eq $CommandArgs -or $CommandArgs.Count -eq 0) {
    & (Join-Path $PSScriptRoot 'avs.ps1')
} else {
    & (Join-Path $PSScriptRoot 'avs.ps1') @CommandArgs
}
exit $LASTEXITCODE
