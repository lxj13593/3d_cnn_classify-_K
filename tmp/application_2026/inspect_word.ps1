$ErrorActionPreference = "Stop"
$docx = "E:\pythonproject\3d_cnn_classify _K\tmp\application_2026\current_ascii.docx"
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
$word.AutomationSecurity = 3
$word.Options.SaveNormalPrompt = $false
try {
    $doc = $word.Documents.Open($docx, $false, $true, $false)
    $pages = $doc.ComputeStatistics(2)
    $words = $doc.ComputeStatistics(0)
    Write-Output ("PAGES=" + $pages)
    Write-Output ("WORDS=" + $words)
    for ($i = 1; $i -le $pages; $i++) {
        $start = $doc.GoTo(1, 1, $i).Start
        if ($i -lt $pages) {
            $end = $doc.GoTo(1, 1, $i + 1).Start - 1
        } else {
            $end = $doc.Content.End
        }
        $pageText = $doc.Range($start, $end).Text
        $compact = [regex]::Replace($pageText, "\s+", "")
        Write-Output ("PAGE_" + $i + "_CHARS=" + $compact.Length)
    }
    $doc.Close($false)
} finally {
    $word.Quit()
}
