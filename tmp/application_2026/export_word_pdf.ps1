$ErrorActionPreference = "Stop"
$docx = "E:\pythonproject\3d_cnn_classify _K\tmp\application_2026\final_ascii.docx"
$renderDir = "E:\pythonproject\3d_cnn_classify _K\tmp\application_2026\final_render_final"
$pdf = Join-Path $renderDir "final.pdf"
New-Item -ItemType Directory -Force -Path $renderDir | Out-Null
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
$word.AutomationSecurity = 3
$word.Options.SaveNormalPrompt = $false
$word.Options.UpdateLinksAtOpen = $false
$word.Options.CheckGrammarAsYouType = $false
$word.Options.CheckSpellingAsYouType = $false
try {
    $doc = $word.Documents.Open($docx, $false, $true, $false)
    $doc.SaveAs2($pdf, 17)
    $doc.Close($false)
} finally {
    $word.Quit()
}
Get-Item -LiteralPath $pdf | Select-Object FullName, Length, LastWriteTime
