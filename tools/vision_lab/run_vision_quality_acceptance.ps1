[CmdletBinding()]
param(
    [string]$OutputDir = "artifacts\vision_lab\v2-2-c0-v1-01",
    [string]$CoppeliaRoot = $(if ($env:COPPELIASIM_ROOT) {
        $env:COPPELIASIM_ROOT
    } else {
        "E:\CoppeliaSim"
    }),
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 23005,
    [ValidateRange(1, 86400)]
    [int]$OnlineTimeoutSeconds = 240,
    [ValidateRange(1, 86400)]
    [int]$ExperimentTimeoutSeconds = 240
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not [System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir = Join-Path $ProjectRoot $OutputDir
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$OutputDir = (Resolve-Path -LiteralPath $OutputDir).Path

$SummaryPath = Join-Path $OutputDir "acceptance-summary.json"
$JUnitPath = Join-Path $OutputDir "vision-quality-online.xml"
$SummaryTempPath = Join-Path $OutputDir (
    ".acceptance-summary." + [guid]::NewGuid().ToString("N") + ".tmp"
)
foreach ($OwnedEvidencePath in @($SummaryPath, $JUnitPath)) {
    if (Test-Path -LiteralPath $OwnedEvidencePath) {
        if (-not (Test-Path -LiteralPath $OwnedEvidencePath -PathType Leaf)) {
            throw "Acceptance evidence target is not a file: $OwnedEvidencePath"
        }
        Remove-Item -LiteralPath $OwnedEvidencePath -Force
    }
}

. (Join-Path $PSScriptRoot "process_ownership.ps1")
$Python = Join-Path $ProjectRoot ".venv-vision\Scripts\python.exe"
$Scene = Join-Path `
    $ProjectRoot `
    "simulation\vision_quality_lab\BL23_vision_quality_lab.ttt"
$CallerQtPlatformExists = Test-Path Env:QT_QPA_PLATFORM
$CallerQtPlatform = $env:QT_QPA_PLATFORM
$CallerPythonUtf8Exists = Test-Path Env:PYTHONUTF8
$CallerPythonUtf8 = $env:PYTHONUTF8
$CallerPythonIoEncodingExists = Test-Path Env:PYTHONIOENCODING
$CallerPythonIoEncoding = $env:PYTHONIOENCODING
$LocationPushed = $false
$OwnedProcessId = $null
$OwnedProcessPath = $null
$OwnedProcessStartTimeUtcTicks = $null
$FailureMessage = $null
$Steps = [ordered]@{}
$ExperimentPayload = $null

if (-not ("RobotSim.KillOnCloseJob" -as [type])) {
    Add-Type -TypeDefinition @"
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace RobotSim
{
    public sealed class KillOnCloseJob : IDisposable
    {
        private const uint KillOnJobClose = 0x00002000;
        private const uint CreateSuspended = 0x00000004;
        private const uint CreateNoWindow = 0x08000000;
        private const uint StartfUseStdHandles = 0x00000100;
        private const uint HandleFlagInherit = 0x00000001;
        private const uint InvalidResumeResult = 0xffffffff;
        private const uint GenericRead = 0x80000000;
        private const uint FileShareRead = 0x00000001;
        private const uint FileShareWrite = 0x00000002;
        private const uint OpenExisting = 3;
        private static readonly IntPtr InvalidHandleValue = new IntPtr(-1);
        private IntPtr handle;

        [StructLayout(LayoutKind.Sequential)]
        private struct BasicLimitInformation
        {
            public long PerProcessUserTimeLimit;
            public long PerJobUserTimeLimit;
            public uint LimitFlags;
            public UIntPtr MinimumWorkingSetSize;
            public UIntPtr MaximumWorkingSetSize;
            public uint ActiveProcessLimit;
            public UIntPtr Affinity;
            public uint PriorityClass;
            public uint SchedulingClass;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct IoCounters
        {
            public ulong ReadOperationCount;
            public ulong WriteOperationCount;
            public ulong OtherOperationCount;
            public ulong ReadTransferCount;
            public ulong WriteTransferCount;
            public ulong OtherTransferCount;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct ExtendedLimitInformation
        {
            public BasicLimitInformation BasicLimitInformation;
            public IoCounters IoInfo;
            public UIntPtr ProcessMemoryLimit;
            public UIntPtr JobMemoryLimit;
            public UIntPtr PeakProcessMemoryUsed;
            public UIntPtr PeakJobMemoryUsed;
        }

        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
        private struct StartupInformation
        {
            public int cb;
            public string lpReserved;
            public string lpDesktop;
            public string lpTitle;
            public uint dwX;
            public uint dwY;
            public uint dwXSize;
            public uint dwYSize;
            public uint dwXCountChars;
            public uint dwYCountChars;
            public uint dwFillAttribute;
            public uint dwFlags;
            public ushort wShowWindow;
            public ushort cbReserved2;
            public IntPtr lpReserved2;
            public IntPtr hStdInput;
            public IntPtr hStdOutput;
            public IntPtr hStdError;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct ProcessInformation
        {
            public IntPtr hProcess;
            public IntPtr hThread;
            public uint dwProcessId;
            public uint dwThreadId;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct SecurityAttributes
        {
            public int nLength;
            public IntPtr lpSecurityDescriptor;
            [MarshalAs(UnmanagedType.Bool)]
            public bool bInheritHandle;
        }

        [DllImport(
            "kernel32.dll",
            CharSet = CharSet.Unicode,
            SetLastError = true)]
        private static extern IntPtr CreateJobObject(
            IntPtr securityAttributes,
            string name
        );

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool SetInformationJobObject(
            IntPtr job,
            int informationClass,
            ref ExtendedLimitInformation information,
            uint informationLength
        );

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool AssignProcessToJobObject(
            IntPtr job,
            IntPtr process
        );

        [DllImport(
            "kernel32.dll",
            CharSet = CharSet.Unicode,
            SetLastError = true)]
        private static extern bool CreateProcess(
            string applicationName,
            StringBuilder commandLine,
            IntPtr processAttributes,
            IntPtr threadAttributes,
            bool inheritHandles,
            uint creationFlags,
            IntPtr environment,
            string currentDirectory,
            ref StartupInformation startupInformation,
            out ProcessInformation processInformation
        );

        [DllImport(
            "kernel32.dll",
            CharSet = CharSet.Unicode,
            SetLastError = true)]
        private static extern IntPtr CreateFile(
            string fileName,
            uint desiredAccess,
            uint shareMode,
            ref SecurityAttributes securityAttributes,
            uint creationDisposition,
            uint flagsAndAttributes,
            IntPtr templateFile
        );

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool SetHandleInformation(
            IntPtr objectHandle,
            uint mask,
            uint flags
        );

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern uint ResumeThread(IntPtr thread);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool TerminateProcess(
            IntPtr process,
            uint exitCode
        );

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool CloseHandle(IntPtr handle);

        public KillOnCloseJob()
        {
            handle = CreateJobObject(IntPtr.Zero, null);
            if (handle == IntPtr.Zero)
            {
                throw new Win32Exception(Marshal.GetLastWin32Error());
            }
            ExtendedLimitInformation information =
                new ExtendedLimitInformation();
            information.BasicLimitInformation.LimitFlags = KillOnJobClose;
            if (!SetInformationJobObject(
                handle,
                9,
                ref information,
                (uint)Marshal.SizeOf(typeof(ExtendedLimitInformation))))
            {
                int error = Marshal.GetLastWin32Error();
                CloseHandle(handle);
                handle = IntPtr.Zero;
                throw new Win32Exception(error);
            }
        }

        private static void MakeInheritable(IntPtr streamHandle)
        {
            if (!SetHandleInformation(
                streamHandle,
                HandleFlagInherit,
                HandleFlagInherit))
            {
                throw new Win32Exception(Marshal.GetLastWin32Error());
            }
        }

        public Process StartSuspended(
            string executable,
            string arguments,
            string workingDirectory,
            string standardOutputPath,
            string standardErrorPath)
        {
            if (handle == IntPtr.Zero)
            {
                throw new ObjectDisposedException("KillOnCloseJob");
            }
            if (String.IsNullOrWhiteSpace(executable))
            {
                throw new ArgumentException("Executable is required.", "executable");
            }
            ProcessInformation processInformation = new ProcessInformation();
            bool created = false;
            bool assigned = false;
            Process managedProcess = null;
            SecurityAttributes security = new SecurityAttributes();
            security.nLength = Marshal.SizeOf(typeof(SecurityAttributes));
            security.bInheritHandle = true;
            IntPtr inputHandle = CreateFile(
                "NUL",
                GenericRead,
                FileShareRead | FileShareWrite,
                ref security,
                OpenExisting,
                0,
                IntPtr.Zero);
            if (inputHandle == InvalidHandleValue)
            {
                throw new Win32Exception(Marshal.GetLastWin32Error());
            }
            try
            {
                using (FileStream standardOutput = new FileStream(
                    standardOutputPath,
                    FileMode.Create,
                    FileAccess.Write,
                    FileShare.ReadWrite))
                using (FileStream standardError = new FileStream(
                    standardErrorPath,
                    FileMode.Create,
                    FileAccess.Write,
                    FileShare.ReadWrite))
                {
                IntPtr outputHandle =
                    standardOutput.SafeFileHandle.DangerousGetHandle();
                IntPtr errorHandle =
                    standardError.SafeFileHandle.DangerousGetHandle();
                MakeInheritable(outputHandle);
                MakeInheritable(errorHandle);
                StartupInformation startup = new StartupInformation();
                startup.cb = Marshal.SizeOf(typeof(StartupInformation));
                startup.dwFlags = StartfUseStdHandles;
                startup.hStdInput = inputHandle;
                startup.hStdOutput = outputHandle;
                startup.hStdError = errorHandle;
                StringBuilder commandLine = new StringBuilder();
                commandLine.Append('"');
                commandLine.Append(executable.Replace("\"", "\\\""));
                commandLine.Append('"');
                if (!String.IsNullOrEmpty(arguments))
                {
                    commandLine.Append(' ');
                    commandLine.Append(arguments);
                }
                try
                {
                    created = CreateProcess(
                        executable,
                        commandLine,
                        IntPtr.Zero,
                        IntPtr.Zero,
                        true,
                        CreateSuspended | CreateNoWindow,
                        IntPtr.Zero,
                        workingDirectory,
                        ref startup,
                        out processInformation);
                    if (!created)
                    {
                        throw new Win32Exception(Marshal.GetLastWin32Error());
                    }
                    if (!AssignProcessToJobObject(
                        handle,
                        processInformation.hProcess))
                    {
                        throw new Win32Exception(Marshal.GetLastWin32Error());
                    }
                    assigned = true;
                    managedProcess = Process.GetProcessById(
                        (int)processInformation.dwProcessId);
                    if (ResumeThread(processInformation.hThread) ==
                        InvalidResumeResult)
                    {
                        throw new Win32Exception(Marshal.GetLastWin32Error());
                    }
                    return managedProcess;
                }
                catch
                {
                    if (managedProcess != null)
                    {
                        managedProcess.Dispose();
                    }
                    if (created && !assigned)
                    {
                        TerminateProcess(processInformation.hProcess, 1);
                    }
                    if (assigned)
                    {
                        Dispose();
                    }
                    throw;
                }
                finally
                {
                    if (processInformation.hThread != IntPtr.Zero)
                    {
                        CloseHandle(processInformation.hThread);
                    }
                    if (processInformation.hProcess != IntPtr.Zero)
                    {
                        CloseHandle(processInformation.hProcess);
                    }
                }
                }
            }
            finally
            {
                CloseHandle(inputHandle);
            }
        }

        public void Dispose()
        {
            if (handle == IntPtr.Zero)
            {
                return;
            }
            IntPtr current = handle;
            handle = IntPtr.Zero;
            if (!CloseHandle(current))
            {
                throw new Win32Exception(Marshal.GetLastWin32Error());
            }
        }
    }
}
"@
}

function ConvertTo-NativeArgument {
    param([AllowEmptyString()][string]$Value)

    if ($Value.Length -gt 0 -and $Value -notmatch '[\s"]') {
        return $Value
    }
    $Builder = New-Object System.Text.StringBuilder
    [void]$Builder.Append([char]34)
    [int]$Backslashes = 0
    foreach ($Character in $Value.ToCharArray()) {
        if ($Character -eq [char]92) {
            $Backslashes += 1
            continue
        }
        if ($Character -eq [char]34) {
            [void]$Builder.Append(('\' * (($Backslashes * 2) + 1)))
            [void]$Builder.Append([char]34)
            $Backslashes = 0
            continue
        }
        if ($Backslashes -gt 0) {
            [void]$Builder.Append(('\' * $Backslashes))
            $Backslashes = 0
        }
        [void]$Builder.Append($Character)
    }
    if ($Backslashes -gt 0) {
        [void]$Builder.Append(('\' * ($Backslashes * 2)))
    }
    [void]$Builder.Append([char]34)
    return $Builder.ToString()
}

function Remove-BoundedTemporaryFile {
    param(
        [string]$Path,
        [int]$TimeoutMilliseconds = 5000
    )

    $Deadline = [DateTime]::UtcNow.AddMilliseconds($TimeoutMilliseconds)
    while (Test-Path -LiteralPath $Path) {
        try {
            Remove-Item -LiteralPath $Path -Force -ErrorAction Stop
            return
        } catch {
            if ([DateTime]::UtcNow -ge $Deadline) {
                throw "Could not remove owned temporary output: $Path"
            }
            Start-Sleep -Milliseconds 50
        }
    }
}

function Invoke-BoundedPythonProcess {
    param(
        [string]$Name,
        [string[]]$Arguments,
        [int]$TimeoutSeconds
    )

    $Token = [guid]::NewGuid().ToString("N")
    $StdOutPath = Join-Path $OutputDir ".$Name.$Token.stdout.tmp"
    $StdErrPath = Join-Path $OutputDir ".$Name.$Token.stderr.tmp"
    $StartedProcess = $null
    $ProcessJob = $null
    $AssignedToJob = $false
    $TimedOut = $false
    try {
        $ArgumentLine = (
            $Arguments |
            ForEach-Object { ConvertTo-NativeArgument ([string]$_) }
        ) -join " "
        $ProcessJob = New-Object RobotSim.KillOnCloseJob
        $StartedProcess = $ProcessJob.StartSuspended(
            $Python,
            $ArgumentLine,
            $ProjectRoot,
            $StdOutPath,
            $StdErrPath
        )
        $AssignedToJob = $true
        Wait-Process `
            -InputObject $StartedProcess `
            -Timeout $TimeoutSeconds `
            -ErrorAction SilentlyContinue
        $StartedProcess.Refresh()
        if (-not $StartedProcess.HasExited) {
            $TimedOut = $true
        }
        $ProcessJob.Dispose()
        $ProcessJob = $null
        Wait-Process `
            -InputObject $StartedProcess `
            -Timeout 10 `
            -ErrorAction SilentlyContinue
        $StartedProcess.Refresh()
        if (-not $StartedProcess.HasExited) {
            Stop-StartedProcessObject -StartedProcess $StartedProcess
            throw "Python process tree did not exit after job close: $Name"
        }
        $StdOutText = if (Test-Path -LiteralPath $StdOutPath -PathType Leaf) {
            [string](Get-Content -LiteralPath $StdOutPath -Raw -Encoding UTF8)
        } else {
            ""
        }
        $StdErrText = if (Test-Path -LiteralPath $StdErrPath -PathType Leaf) {
            [string](Get-Content -LiteralPath $StdErrPath -Raw -Encoding UTF8)
        } else {
            ""
        }
        if (-not [string]::IsNullOrWhiteSpace($StdOutText)) {
            Write-Host $StdOutText.TrimEnd()
        }
        if (-not [string]::IsNullOrWhiteSpace($StdErrText)) {
            Write-Host $StdErrText.TrimEnd()
        }
        return [pscustomobject]@{
            ExitCode = $(if ($TimedOut) { $null } else { $StartedProcess.ExitCode })
            TimedOut = $TimedOut
            StdOut = $StdOutText
            StdErr = $StdErrText
        }
    } finally {
        try {
            if ($null -ne $ProcessJob) {
                $ProcessJob.Dispose()
            }
        } finally {
            if (
                -not $AssignedToJob -and
                $null -ne $StartedProcess -and
                -not $StartedProcess.HasExited
            ) {
                Stop-StartedProcessObject -StartedProcess $StartedProcess
            }
            if ($null -ne $StartedProcess) {
                $StartedProcess.Dispose()
                $StartedProcess = $null
            }
            foreach ($TemporaryPath in @($StdOutPath, $StdErrPath)) {
                Remove-BoundedTemporaryFile -Path $TemporaryPath
            }
        }
    }
}

function Invoke-CheckedPython {
    param(
        [string]$Name,
        [string[]]$Arguments,
        [int]$TimeoutSeconds
    )
    $StartedAt = Get-Date
    $Result = Invoke-BoundedPythonProcess `
        -Name $Name `
        -Arguments $Arguments `
        -TimeoutSeconds $TimeoutSeconds
    $ExitCode = $Result.ExitCode
    $Steps[$Name] = [ordered]@{
        status = $(
            if (-not $Result.TimedOut -and $ExitCode -eq 0) {
                "PASS"
            } else {
                "FAIL"
            }
        )
        exit_code = $ExitCode
        timed_out = $Result.TimedOut
        timeout_seconds = $TimeoutSeconds
        elapsed_seconds = [math]::Round(
            ((Get-Date) - $StartedAt).TotalSeconds,
            3
        )
        command = "python " + ($Arguments -join " ")
    }
    if ($Result.TimedOut) {
        throw "Step '$Name' timed out after $TimeoutSeconds seconds"
    }
    if ($ExitCode -ne 0) {
        throw "Step '$Name' failed with exit code $ExitCode"
    }
}

function Assert-JUnitNoSkips {
    param(
        [string]$Name,
        [string]$Path,
        [int]$ExpectedTests
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        $Steps[$Name]["status"] = "FAIL"
        throw "Step '$Name' did not produce JUnit evidence: $Path"
    }
    [xml]$Report = Get-Content -LiteralPath $Path -Raw -Encoding UTF8
    $Suites = @($Report.SelectNodes("//testsuite"))
    if ($Suites.Count -eq 0) {
        $Steps[$Name]["status"] = "FAIL"
        throw "Step '$Name' produced invalid JUnit evidence: $Path"
    }
    [int]$Tests = 0
    [int]$Skipped = 0
    [int]$Failures = 0
    [int]$Errors = 0
    foreach ($Suite in $Suites) {
        $Tests += [int]$Suite.tests
        $Skipped += [int]$Suite.skipped
        $Failures += [int]$Suite.failures
        $Errors += [int]$Suite.errors
    }
    $Steps[$Name]["tests"] = $Tests
    $Steps[$Name]["skipped"] = $Skipped
    $Steps[$Name]["failures"] = $Failures
    $Steps[$Name]["errors"] = $Errors
    if (
        $Tests -ne $ExpectedTests -or
        $Skipped -ne 0 -or
        $Failures -ne 0 -or
        $Errors -ne 0
    ) {
        $Steps[$Name]["status"] = "FAIL"
        throw (
            "Step '$Name' requires exactly $ExpectedTests tests and zero " +
            "skips, failures and errors; tests=$Tests skipped=$Skipped " +
            "failures=$Failures errors=$Errors"
        )
    }
}

try {
    Remove-Item Env:QT_QPA_PLATFORM -ErrorAction SilentlyContinue
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"
    Push-Location -LiteralPath $ProjectRoot
    $LocationPushed = $true
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        throw "Python environment is missing: $Python"
    }
    if (-not (Test-Path -LiteralPath $Scene -PathType Leaf)) {
        throw "Vision quality scene is missing: $Scene"
    }
    $ExistingListener = Get-NetTCPConnection `
        -LocalPort $Port `
        -State Listen `
        -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($ExistingListener) {
        throw "Dedicated CoppeliaSim port $Port is already occupied"
    }

    $Launch = & (Join-Path $PSScriptRoot "launch_coppeliasim.ps1") `
        -CoppeliaRoot $CoppeliaRoot `
        -Scene $Scene `
        -HostAddress $HostAddress `
        -Port $Port `
        -Hidden
    if (-not $Launch.StartedByScript) {
        throw "Vision quality acceptance did not own the CoppeliaSim process"
    }
    $OwnedProcessId = [int]$Launch.ProcessId
    $OwnedProcessPath = [string]$Launch.ProcessPath
    $OwnedProcessStartTimeUtcTicks = [long](
        $Launch.ProcessStartTimeUtcTicks
    )

    Invoke-CheckedPython `
        -Name "vision_quality_online" `
        -TimeoutSeconds $OnlineTimeoutSeconds `
        -Arguments @(
            "-m", "pytest",
            "tests/test_acceptance/test_coppeliasim_vision_quality_scene.py",
            "tests/test_acceptance/test_coppeliasim_v1_01.py",
            "-m", "coppeliasim",
            "--coppelia-host", $HostAddress,
            "--coppelia-port", [string]$Port,
            "--junitxml", $JUnitPath,
            "-q"
        )
    Assert-JUnitNoSkips `
        -Name "vision_quality_online" `
        -Path $JUnitPath `
        -ExpectedTests 2

    $ExperimentOutput = Join-Path $OutputDir "experiment-runs"
    $ExperimentArguments = @(
        "-m", "vision_platform.cli", "experiment-run",
        "--experiment", "V1-01",
        "--host", $HostAddress,
        "--port", [string]$Port,
        "--output", $ExperimentOutput
    )
    $ExperimentStartedAt = Get-Date
    $ExperimentResult = Invoke-BoundedPythonProcess `
        -Name "v1_01_experiment_run" `
        -Arguments $ExperimentArguments `
        -TimeoutSeconds $ExperimentTimeoutSeconds
    $ExperimentExitCode = $ExperimentResult.ExitCode
    $ExperimentText = $ExperimentResult.StdOut.Trim()
    if ($ExperimentResult.TimedOut) {
        throw (
            "V1-01 experiment-run timed out after " +
            "$ExperimentTimeoutSeconds seconds"
        )
    }
    try {
        $ExperimentPayload = $ExperimentText | ConvertFrom-Json
    } catch {
        throw "V1-01 experiment-run did not return valid JSON"
    }
    $ExperimentPassed = (
        $ExperimentExitCode -eq 0 -and
        $ExperimentPayload.status -eq "PASS" -and
        $ExperimentPayload.hardware_status -eq "PENDING_HARDWARE"
    )
    $Steps["v1_01_experiment_run"] = [ordered]@{
        status = $(if ($ExperimentPassed) { "PASS" } else { "FAIL" })
        exit_code = $ExperimentExitCode
        elapsed_seconds = [math]::Round(
            ((Get-Date) - $ExperimentStartedAt).TotalSeconds,
            3
        )
        command = "python " + ($ExperimentArguments -join " ")
        summary = $ExperimentPayload.summary
        evidence = $ExperimentPayload.evidence
    }
    if (-not $ExperimentPassed) {
        throw "V1-01 experiment-run failed its PASS/PENDING_HARDWARE contract"
    }
} catch {
    $FailureMessage = $_.Exception.Message
} finally {
    if ($LocationPushed) {
        try {
            Pop-Location
        } catch {
            $LocationFailure = "Location restore failed: " + $_.Exception.Message
            $FailureMessage = if ($FailureMessage) {
                $FailureMessage + [Environment]::NewLine + $LocationFailure
            } else {
                $LocationFailure
            }
        }
    }
    if ($OwnedProcessId) {
        try {
            Stop-ExactOwnedProcess `
                -ProcessId $OwnedProcessId `
                -ProcessPath $OwnedProcessPath `
                -ProcessStartTimeUtcTicks $OwnedProcessStartTimeUtcTicks
            if (
                Get-NetTCPConnection `
                    -LocalPort $Port `
                    -State Listen `
                    -ErrorAction SilentlyContinue
            ) {
                throw "CoppeliaSim listener remains on port $Port"
            }
        } catch {
            $CleanupFailure = (
                "Owned CoppeliaSim cleanup failed: " +
                $_.Exception.Message
            )
            $FailureMessage = if ($FailureMessage) {
                $FailureMessage + [Environment]::NewLine + $CleanupFailure
            } else {
                $CleanupFailure
            }
        }
    }
    if ($CallerQtPlatformExists) {
        $env:QT_QPA_PLATFORM = $CallerQtPlatform
    } else {
        Remove-Item Env:QT_QPA_PLATFORM -ErrorAction SilentlyContinue
    }
    if ($CallerPythonUtf8Exists) {
        $env:PYTHONUTF8 = $CallerPythonUtf8
    } else {
        Remove-Item Env:PYTHONUTF8 -ErrorAction SilentlyContinue
    }
    if ($CallerPythonIoEncodingExists) {
        $env:PYTHONIOENCODING = $CallerPythonIoEncoding
    } else {
        Remove-Item Env:PYTHONIOENCODING -ErrorAction SilentlyContinue
    }

    $OverallStatus = if ($FailureMessage) { "FAIL" } else { "PASS" }
    $Summary = [ordered]@{
        schema_version = 1
        generated_at = (Get-Date).ToUniversalTime().ToString("o")
        status = $OverallStatus
        scene = $Scene
        host = $HostAddress
        port = $Port
        steps = $Steps
        experiment_summary = $(
            if ($ExperimentPayload) { $ExperimentPayload.summary } else { $null }
        )
        experiment_evidence = $(
            if ($ExperimentPayload) { $ExperimentPayload.evidence } else { $null }
        )
        failure = $FailureMessage
        hardware_status = "PENDING_HARDWARE"
        teaching_effect = "PENDING_HUMAN_ACCEPTANCE"
        note = (
            "Hardware and teaching-effect gates are not automated PASS."
        )
    }
    $SummaryJson = $Summary | ConvertTo-Json -Depth 12
    $Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText(
        $SummaryTempPath,
        $SummaryJson + [Environment]::NewLine,
        $Utf8NoBom
    )
    Move-Item -LiteralPath $SummaryTempPath -Destination $SummaryPath
}

Write-Host "Vision quality acceptance summary: $SummaryPath"
if ($FailureMessage) {
    throw $FailureMessage
}
