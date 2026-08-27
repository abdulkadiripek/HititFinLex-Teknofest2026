param(
    [string]$OutputDirectory = (Join-Path (Split-Path -Parent $PSScriptRoot) "docs\teslim"),
    [switch]$SkipVideo
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ScreenshotDirectory = Join-Path $OutputDirectory "ekran-goruntuleri"
$PresentationPath = Join-Path $OutputDirectory "HititFinLex_Teknofest2026_Sunumu.pptx"
$PdfPath = Join-Path $OutputDirectory "HititFinLex_Teknofest2026_Sunumu.pdf"
$DemoSourcePath = Join-Path $OutputDirectory "HititFinLex_Demo_Akisi.pptx"
$DemoVideoPath = Join-Path $OutputDirectory "HititFinLex_Demo_Videosu.mp4"
$ChecksumPath = Join-Path $OutputDirectory "SHA256SUMS.txt"

$Colors = @{
    Navy = "071B2D"
    NavySoft = "0E2A3C"
    Panel = "12364A"
    Teal = "20D0B2"
    TealDark = "0E9F8A"
    Cyan = "75E6D2"
    White = "F7FBFC"
    Muted = "A9C1CB"
    Line = "2B5061"
    Amber = "FFCC66"
    Red = "FF7B7B"
}

function ConvertTo-OfficeRgb {
    param([Parameter(Mandatory)][string]$Hex)
    $clean = $Hex.TrimStart("#")
    $red = [Convert]::ToInt32($clean.Substring(0, 2), 16)
    $green = [Convert]::ToInt32($clean.Substring(2, 2), 16)
    $blue = [Convert]::ToInt32($clean.Substring(4, 2), 16)
    return $red -bor ($green -shl 8) -bor ($blue -shl 16)
}

function Add-Rectangle {
    param(
        [Parameter(Mandatory)]$Slide,
        [double]$Left,
        [double]$Top,
        [double]$Width,
        [double]$Height,
        [string]$Fill = $Colors.Panel,
        [double]$Transparency = 0,
        [switch]$Rounded,
        [string]$Line = ""
    )
    $shapeType = if ($Rounded) { 5 } else { 1 }
    $shape = $Slide.Shapes.AddShape($shapeType, $Left, $Top, $Width, $Height)
    $shape.Fill.ForeColor.RGB = ConvertTo-OfficeRgb $Fill
    $shape.Fill.Transparency = $Transparency
    if ($Line) {
        $shape.Line.Visible = -1
        $shape.Line.ForeColor.RGB = ConvertTo-OfficeRgb $Line
        $shape.Line.Weight = 1
    } else {
        $shape.Line.Visible = 0
    }
    return $shape
}

function Add-Text {
    param(
        [Parameter(Mandatory)]$Slide,
        [Parameter(Mandatory)][string]$Text,
        [double]$Left,
        [double]$Top,
        [double]$Width,
        [double]$Height,
        [double]$Size = 18,
        [string]$Color = $Colors.White,
        [switch]$Bold,
        [int]$Align = 1,
        [string]$Font = "Aptos",
        [double]$LineSpacing = 1.0
    )
    $shape = $Slide.Shapes.AddTextbox(1, $Left, $Top, $Width, $Height)
    $shape.TextFrame2.MarginLeft = 0
    $shape.TextFrame2.MarginRight = 0
    $shape.TextFrame2.MarginTop = 0
    $shape.TextFrame2.MarginBottom = 0
    $shape.TextFrame2.WordWrap = -1
    $range = $shape.TextFrame2.TextRange
    $range.Text = $Text
    $range.Font.Name = $Font
    $range.Font.Size = $Size
    $range.Font.Bold = if ($Bold) { -1 } else { 0 }
    $range.Font.Fill.ForeColor.RGB = ConvertTo-OfficeRgb $Color
    $range.ParagraphFormat.Alignment = $Align
    $range.ParagraphFormat.SpaceWithin = $LineSpacing
    return $shape
}

function Add-BaseSlide {
    param([Parameter(Mandatory)]$Presentation)
    $slide = $Presentation.Slides.Add($Presentation.Slides.Count + 1, 12)
    [void](Add-Rectangle -Slide $slide -Left 0 -Top 0 -Width 960 -Height 540 -Fill $Colors.Navy)
    [void](Add-Rectangle -Slide $slide -Left 0 -Top 0 -Width 7 -Height 540 -Fill $Colors.Teal)
    return $slide
}

function Add-Header {
    param(
        [Parameter(Mandatory)]$Slide,
        [string]$Eyebrow,
        [string]$Title,
        [string]$Subtitle,
        [int]$Number
    )
    [void](Add-Text -Slide $Slide -Text $Eyebrow.ToUpperInvariant() -Left 48 -Top 25 -Width 500 -Height 20 -Size 11 -Color $Colors.Teal -Bold)
    [void](Add-Text -Slide $Slide -Text $Title -Left 48 -Top 49 -Width 820 -Height 48 -Size 30 -Bold)
    if ($Subtitle) {
        [void](Add-Text -Slide $Slide -Text $Subtitle -Left 48 -Top 98 -Width 820 -Height 36 -Size 14 -Color $Colors.Muted)
    }
    [void](Add-Text -Slide $Slide -Text ("{0:00}" -f $Number) -Left 887 -Top 30 -Width 30 -Height 20 -Size 11 -Color $Colors.Muted -Align 3)
}

function Add-Footer {
    param([Parameter(Mandatory)]$Slide)
    [void](Add-Rectangle -Slide $Slide -Left 48 -Top 510 -Width 864 -Height 1 -Fill $Colors.Line)
    [void](Add-Text -Slide $Slide -Text "HititFinLex · TEKNOFEST 2026 · #BilisimVadisi2026" -Left 48 -Top 516 -Width 560 -Height 16 -Size 9 -Color $Colors.Muted)
    [void](Add-Text -Slide $Slide -Text "Finansal tavsiye değildir" -Left 700 -Top 516 -Width 212 -Height 16 -Size 9 -Color $Colors.Muted -Align 3)
}

function Add-MetricCard {
    param(
        [Parameter(Mandatory)]$Slide,
        [string]$Value,
        [string]$Label,
        [double]$Left,
        [double]$Top,
        [double]$Width = 190,
        [string]$Accent = $Colors.Teal
    )
    [void](Add-Rectangle -Slide $Slide -Left $Left -Top $Top -Width $Width -Height 88 -Fill $Colors.Panel -Rounded -Line $Colors.Line)
    [void](Add-Rectangle -Slide $Slide -Left ($Left + 16) -Top ($Top + 14) -Width 32 -Height 4 -Fill $Accent -Rounded)
    [void](Add-Text -Slide $Slide -Text $Value -Left ($Left + 16) -Top ($Top + 25) -Width ($Width - 32) -Height 32 -Size 23 -Bold)
    [void](Add-Text -Slide $Slide -Text $Label -Left ($Left + 16) -Top ($Top + 59) -Width ($Width - 32) -Height 20 -Size 10 -Color $Colors.Muted)
}

function Add-FeatureCard {
    param(
        [Parameter(Mandatory)]$Slide,
        [string]$Index,
        [string]$Title,
        [string]$Body,
        [double]$Left,
        [double]$Top,
        [double]$Width,
        [double]$Height
    )
    [void](Add-Rectangle -Slide $Slide -Left $Left -Top $Top -Width $Width -Height $Height -Fill $Colors.Panel -Rounded -Line $Colors.Line)
    [void](Add-Rectangle -Slide $Slide -Left ($Left + 18) -Top ($Top + 18) -Width 34 -Height 34 -Fill $Colors.TealDark -Rounded)
    [void](Add-Text -Slide $Slide -Text $Index -Left ($Left + 18) -Top ($Top + 25) -Width 34 -Height 18 -Size 12 -Bold -Align 2)
    [void](Add-Text -Slide $Slide -Text $Title -Left ($Left + 66) -Top ($Top + 18) -Width ($Width - 84) -Height 28 -Size 17 -Bold)
    [void](Add-Text -Slide $Slide -Text $Body -Left ($Left + 18) -Top ($Top + 62) -Width ($Width - 36) -Height ($Height - 76) -Size 12 -Color $Colors.Muted -LineSpacing 1.08)
}

function Add-FlowNode {
    param(
        [Parameter(Mandatory)]$Slide,
        [string]$Title,
        [string]$Body,
        [double]$Left,
        [double]$Top,
        [double]$Width = 176,
        [string]$Fill = $Colors.Panel
    )
    [void](Add-Rectangle -Slide $Slide -Left $Left -Top $Top -Width $Width -Height 100 -Fill $Fill -Rounded -Line $Colors.Line)
    [void](Add-Text -Slide $Slide -Text $Title -Left ($Left + 14) -Top ($Top + 18) -Width ($Width - 28) -Height 25 -Size 15 -Bold)
    [void](Add-Text -Slide $Slide -Text $Body -Left ($Left + 14) -Top ($Top + 49) -Width ($Width - 28) -Height 40 -Size 10 -Color $Colors.Muted)
}

function Add-Arrow {
    param(
        [Parameter(Mandatory)]$Slide,
        [double]$FromX,
        [double]$FromY,
        [double]$ToX,
        [double]$ToY
    )
    $arrow = $Slide.Shapes.AddConnector(1, $FromX, $FromY, $ToX, $ToY)
    $arrow.Line.ForeColor.RGB = ConvertTo-OfficeRgb $Colors.Teal
    $arrow.Line.Weight = 2
    $arrow.Line.EndArrowheadStyle = 3
}

function Add-ScreenshotSlide {
    param(
        [Parameter(Mandatory)]$Presentation,
        [int]$Number,
        [string]$Eyebrow,
        [string]$Title,
        [string]$Subtitle,
        [string]$ImagePath,
        [string]$CalloutTitle,
        [string]$CalloutBody
    )
    $slide = Add-BaseSlide -Presentation $Presentation
    Add-Header -Slide $slide -Eyebrow $Eyebrow -Title $Title -Subtitle $Subtitle -Number $Number
    [void](Add-Rectangle -Slide $slide -Left 48 -Top 145 -Width 650 -Height 345 -Fill $Colors.NavySoft -Rounded -Line $Colors.Line)
    if (Test-Path -LiteralPath $ImagePath) {
        [void]$slide.Shapes.AddPicture($ImagePath, 0, -1, 58, 155, 630, 325)
    } else {
        [void](Add-Text -Slide $slide -Text "Ekran görüntüsü üretim sırasında eklenecek" -Left 100 -Top 300 -Width 540 -Height 30 -Size 18 -Color $Colors.Muted -Align 2)
    }
    [void](Add-Rectangle -Slide $slide -Left 718 -Top 145 -Width 194 -Height 345 -Fill $Colors.Panel -Rounded -Line $Colors.Line)
    [void](Add-Rectangle -Slide $slide -Left 738 -Top 169 -Width 38 -Height 5 -Fill $Colors.Teal -Rounded)
    [void](Add-Text -Slide $slide -Text $CalloutTitle -Left 738 -Top 194 -Width 154 -Height 54 -Size 18 -Bold)
    [void](Add-Text -Slide $slide -Text $CalloutBody -Left 738 -Top 265 -Width 154 -Height 190 -Size 12 -Color $Colors.Muted -LineSpacing 1.1)
    Add-Footer -Slide $slide
    return $slide
}

function Set-SlideTiming {
    param([Parameter(Mandatory)]$Slide, [double]$Seconds = 8)
    $Slide.SlideShowTransition.AdvanceOnClick = 0
    $Slide.SlideShowTransition.AdvanceOnTime = -1
    $Slide.SlideShowTransition.AdvanceTime = $Seconds
}

function Set-OpenXmlCoreMetadata {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Title,
        [Parameter(Mandatory)][string]$Subject
    )

    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [System.IO.Compression.ZipFile]::Open(
        $Path,
        [System.IO.Compression.ZipArchiveMode]::Update
    )
    try {
        $entry = $archive.GetEntry("docProps/core.xml")
        if ($null -eq $entry) {
            throw "PPTX çekirdek metadata girdisi bulunamadı: $Path"
        }
        $stream = $entry.Open()
        try {
            $document = [System.Xml.XmlDocument]::new()
            $document.PreserveWhitespace = $true
            $document.Load($stream)
        } finally {
            $stream.Dispose()
        }

        $namespace = [System.Xml.XmlNamespaceManager]::new($document.NameTable)
        $namespace.AddNamespace("cp", "http://schemas.openxmlformats.org/package/2006/metadata/core-properties")
        $namespace.AddNamespace("dc", "http://purl.org/dc/elements/1.1/")
        $values = @{
            "dc:creator" = "HititFinLex Takımı"
            "cp:lastModifiedBy" = "HititFinLex Takımı"
            "dc:title" = $Title
            "dc:subject" = $Subject
            "cp:keywords" = "HititFinLex, TEKNOFEST 2026, katılım finansı, NLP, RAG"
            "dc:description" = "HititFinLex Takımı tarafından TEKNOFEST 2026 teslimi için üretildi."
        }
        foreach ($item in $values.GetEnumerator()) {
            $node = $document.SelectSingleNode("//$($item.Key)", $namespace)
            if ($null -eq $node) {
                $prefix, $localName = $item.Key.Split(":", 2)
                $node = $document.CreateElement(
                    $prefix,
                    $localName,
                    $namespace.LookupNamespace($prefix)
                )
                [void]$document.DocumentElement.AppendChild($node)
            }
            $node.InnerText = $item.Value
        }

        $stream = $entry.Open()
        try {
            $stream.SetLength(0)
            $settings = [System.Xml.XmlWriterSettings]::new()
            $settings.Encoding = [System.Text.UTF8Encoding]::new($false)
            $settings.Indent = $false
            $writer = [System.Xml.XmlWriter]::Create($stream, $settings)
            try {
                $document.Save($writer)
            } finally {
                $writer.Dispose()
            }
        } finally {
            $stream.Dispose()
        }
    } finally {
        $archive.Dispose()
    }
}

function New-MainPresentation {
    param([Parameter(Mandatory)]$PowerPoint)
    $presentation = $PowerPoint.Presentations.Add()
    $presentation.PageSetup.SlideWidth = 960
    $presentation.PageSetup.SlideHeight = 540

    $slide = Add-BaseSlide -Presentation $presentation
    [void](Add-Rectangle -Slide $slide -Left 620 -Top 0 -Width 340 -Height 540 -Fill $Colors.TealDark)
    [void](Add-Rectangle -Slide $slide -Left 665 -Top 85 -Width 190 -Height 190 -Fill $Colors.Navy -Rounded)
    [void](Add-Text -Slide $slide -Text "H" -Left 665 -Top 104 -Width 190 -Height 140 -Size 96 -Bold -Align 2)
    [void](Add-Text -Slide $slide -Text "HITITFINLEX" -Left 55 -Top 70 -Width 500 -Height 24 -Size 13 -Color $Colors.Teal -Bold)
    [void](Add-Text -Slide $slide -Text "Katılım finansı için`nkaynaklı karar desteği" -Left 55 -Top 108 -Width 530 -Height 135 -Size 39 -Bold)
    [void](Add-Text -Slide $slide -Text "Güncel ve tarihsel banka ürünlerini keşfedin, karşılaştırın ve her sonucu resmî kaynağına kadar izleyin." -Left 55 -Top 260 -Width 500 -Height 65 -Size 17 -Color $Colors.Muted -LineSpacing 1.05)
    [void](Add-Text -Slide $slide -Text "TEKNOFEST 2026 · Yapay Zekâ Dil Ajanları · 2. Senaryo" -Left 55 -Top 355 -Width 500 -Height 25 -Size 12 -Color $Colors.Cyan -Bold)
    [void](Add-Text -Slide $slide -Text "Emre Deniz · Doğukan Ayas · Tuğba Melisa Güngör Kurnaz · Abdulkadir İpek" -Left 55 -Top 390 -Width 520 -Height 45 -Size 11 -Color $Colors.Muted)
    [void](Add-Text -Slide $slide -Text "#BilisimVadisi2026" -Left 665 -Top 405 -Width 190 -Height 24 -Size 13 -Color $Colors.White -Bold -Align 2)
    [void](Add-Text -Slide $slide -Text "01" -Left 887 -Top 500 -Width 30 -Height 18 -Size 10 -Color $Colors.Navy -Bold -Align 3)

    $slide = Add-BaseSlide -Presentation $presentation
    Add-Header -Slide $slide -Eyebrow "Problem ve fırsat" -Title "Dağınık bilgi, zor karşılaştırma, zayıf izlenebilirlik" -Subtitle "Katılım bankacılığı ürün koşulları farklı sayfalarda, farklı anlatımlarda ve zaman içinde değişerek yayımlanıyor." -Number 2
    Add-FeatureCard -Slide $slide -Index "01" -Title "Parçalı kaynaklar" -Body "Kampanya, finansman ve hesap koşulları bankaların farklı sayfalarına dağılmış durumda." -Left 48 -Top 165 -Width 270 -Height 210
    Add-FeatureCard -Slide $slide -Index "02" -Title "Standart olmayan dil" -Body "Tutar, vade, oran, ücret ve ödül bilgileri aynı anlamı taşısa da farklı biçimlerde yazılıyor." -Left 345 -Top 165 -Width 270 -Height 210
    Add-FeatureCard -Slide $slide -Index "03" -Title "Kaynak ihtiyacı" -Body "Karar destek çıktısının hangi belge ve kanıt cümlesinden üretildiği açıkça görülebilmeli." -Left 642 -Top 165 -Width 270 -Height 210
    [void](Add-Rectangle -Slide $slide -Left 48 -Top 400 -Width 864 -Height 82 -Fill $Colors.TealDark -Rounded)
    [void](Add-Text -Slide $slide -Text "Çözüm: güncel + tarihsel belgeleri tek veri katmanında birleştiren, finansal alanları yapılandıran ve her çıktıyı kaynakla sunan yerel bir NLP/RAG platformu." -Left 72 -Top 421 -Width 816 -Height 42 -Size 16 -Bold -Align 2)
    Add-Footer -Slide $slide

    $slide = Add-BaseSlide -Presentation $presentation
    Add-Header -Slide $slide -Eyebrow "Ürün" -Title "Tek platformda beş karar destek görünümü" -Subtitle "Arama, karşılaştırma ve yapay zekâ çıktıları aynı kaynak ve kalite sözleşmesini paylaşır." -Number 3
    Add-FeatureCard -Slide $slide -Index "01" -Title "Genel bakış" -Body "Kapsam, banka, belge, çıkarım ve sistem hazırlık göstergeleri." -Left 48 -Top 155 -Width 260 -Height 150
    Add-FeatureCard -Slide $slide -Index "02" -Title "Akıllı katalog" -Body "Banka, ürün, tarih, güven ve bilgi kapsamı filtreleri." -Left 326 -Top 155 -Width 260 -Height 150
    Add-FeatureCard -Slide $slide -Index "03" -Title "Karşılaştırma" -Body "Banka koşullarını kanıt ve güven bilgisiyle yan yana sunan matris." -Left 604 -Top 155 -Width 308 -Height 150
    Add-FeatureCard -Slide $slide -Index "04" -Title "Kaynaklı asistan" -Body "BGE-M3 hibrit arama + yerel Qwen ile dipnotlu yanıt." -Left 48 -Top 325 -Width 399 -Height 150
    Add-FeatureCard -Slide $slide -Index "05" -Title "Veri kalitesi" -Body "Silver/model çıkarımı, güven, kapsam ve insan inceleme kuyruğu." -Left 465 -Top 325 -Width 447 -Height 150
    Add-Footer -Slide $slide

    $slide = Add-BaseSlide -Presentation $presentation
    Add-Header -Slide $slide -Eyebrow "Mimari" -Title "Kaynak metinden açıklanabilir yanıta" -Subtitle "Bulut LLM bağımlılığı olmayan, PostgreSQL ve yerel modeller üzerinde çalışan uçtan uca akış." -Number 4
    Add-FlowNode -Slide $slide -Title "Resmî kaynaklar" -Body "Banka siteleri + Internet Archive" -Left 48 -Top 190 -Width 150
    Add-Arrow -Slide $slide -FromX 198 -FromY 240 -ToX 222 -ToY 240
    Add-FlowNode -Slide $slide -Title "NLP hattı" -Body "BERT sınıflandırma + NER + bağlam kuralları" -Left 222 -Top 190 -Width 150
    Add-Arrow -Slide $slide -FromX 372 -FromY 240 -ToX 396 -ToY 240
    Add-FlowNode -Slide $slide -Title "Veri katmanı" -Body "PostgreSQL 18 + pgvector + tarihsel arşiv" -Left 396 -Top 190 -Width 150
    Add-Arrow -Slide $slide -FromX 546 -FromY 240 -ToX 570 -ToY 240
    Add-FlowNode -Slide $slide -Title "RAG ve API" -Body "BGE-M3 arama + Qwen + FastAPI" -Left 570 -Top 190 -Width 150
    Add-Arrow -Slide $slide -FromX 720 -FromY 240 -ToX 744 -ToY 240
    Add-FlowNode -Slide $slide -Title "Next.js arayüz" -Body "Katalog + karşılaştırma + asistan" -Left 744 -Top 190 -Width 168 -Fill $Colors.TealDark
    [void](Add-Rectangle -Slide $slide -Left 48 -Top 335 -Width 864 -Height 110 -Fill $Colors.NavySoft -Rounded -Line $Colors.Line)
    [void](Add-Text -Slide $slide -Text "Tasarım ilkeleri" -Left 70 -Top 355 -Width 180 -Height 25 -Size 17 -Bold)
    [void](Add-Text -Slide $slide -Text "• Kaynak cümlesine kadar izlenebilirlik`n• Güncel ve tarihsel kapsamın ayrıştırılması`n• Doğrulanmamış model çıktısının açık etiketi" -Left 265 -Top 351 -Width 300 -Height 75 -Size 12 -Color $Colors.Muted)
    [void](Add-Text -Slide $slide -Text "• Yerel GPU ve yerel LLM`n• İdempotent belge alımı ve inceleme akışı`n• Güvenli public/admin API ayrımı" -Left 590 -Top 351 -Width 285 -Height 75 -Size 12 -Color $Colors.Muted)
    Add-Footer -Slide $slide

    $slide = Add-BaseSlide -Presentation $presentation
    Add-Header -Slide $slide -Eyebrow "Veri seti" -Title "2016–2026 dönemini kapsayan sürümlü finans korpusu" -Subtitle "Resmî paket JSONL/CSV, şema, manifest ve veri kartıyla birlikte yayımlanır." -Number 5
    Add-MetricCard -Slide $slide -Value "3.351" -Label "toplam belge" -Left 48 -Top 160 -Width 195
    Add-MetricCard -Slide $slide -Value "771" -Label "güncel belge" -Left 263 -Top 160 -Width 195
    Add-MetricCard -Slide $slide -Value "2.580" -Label "tarihsel belge" -Left 478 -Top 160 -Width 195
    Add-MetricCard -Slide $slide -Value "10.501" -Label "etiketli span" -Left 693 -Top 160 -Width 219
    Add-MetricCard -Slide $slide -Value "10" -Label "katılım bankası" -Left 48 -Top 270 -Width 195 -Accent $Colors.Cyan
    Add-MetricCard -Slide $slide -Value "7.238" -Label "NER pasajı" -Left 263 -Top 270 -Width 195 -Accent $Colors.Cyan
    Add-MetricCard -Slide $slide -Value "1,04 M" -Label "kelime" -Left 478 -Top 270 -Width 195 -Accent $Colors.Cyan
    Add-MetricCard -Slide $slide -Value "2016–2026" -Label "içerik tarih aralığı" -Left 693 -Top 270 -Width 219 -Accent $Colors.Cyan
    [void](Add-Rectangle -Slide $slide -Left 48 -Top 395 -Width 864 -Height 82 -Fill "4A3B1A" -Rounded -Line $Colors.Amber)
    [void](Add-Text -Slide $slide -Text "VERİ YÖNETİŞİMİ" -Left 70 -Top 412 -Width 165 -Height 18 -Size 10 -Color $Colors.Amber -Bold)
    [void](Add-Text -Slide $slide -Text "Etiketler silver/kural tabanlıdır; tüm `doğrulandi` alanları insan doğrulaması tamamlanana kadar false kalır. Arayüz ve API bunu model çıkarımı olarak açıkça gösterir." -Left 235 -Top 407 -Width 645 -Height 52 -Size 13 -Color $Colors.White)
    Add-Footer -Slide $slide

    $slide = Add-BaseSlide -Presentation $presentation
    Add-Header -Slide $slide -Eyebrow "NLP ve RAG" -Title "Türkçe finans diline özel hibrit yaklaşım" -Subtitle "Öğrenen modeller, deterministik bağlam kuralları ve kaynaklı üretim birlikte çalışır." -Number 6
    Add-FeatureCard -Slide $slide -Index "01" -Title "Sınıflandırma" -Body "BERTurk tabanlı kampanya ve 13 sınıflı ürün modeli. Ürün sınıflandırıcı Macro F1: 0,8759." -Left 48 -Top 160 -Width 270 -Height 200
    Add-FeatureCard -Slide $slide -Index "02" -Title "Varlık çıkarımı" -Body "17 finansal varlık türü. NER Precision 0,8296 · Recall 0,9113 · F1 0,8685." -Left 345 -Top 160 -Width 270 -Height 200
    Add-FeatureCard -Slide $slide -Index "03" -Title "Kural + RAG" -Body "Bağlam güvenlik kuralları; BGE-M3 dense+sparse arama; Qwen ile yalnızca getirilen kaynaklardan yanıt." -Left 642 -Top 160 -Width 270 -Height 200
    [void](Add-Rectangle -Slide $slide -Left 48 -Top 390 -Width 864 -Height 80 -Fill $Colors.NavySoft -Rounded -Line $Colors.Line)
    [void](Add-Text -Slide $slide -Text "Model → bağlam kuralı → güven eşiği → insan inceleme kuyruğu" -Left 80 -Top 411 -Width 800 -Height 35 -Size 20 -Bold -Align 2)
    Add-Footer -Slide $slide

    [void](Add-ScreenshotSlide -Presentation $presentation -Number 7 -Eyebrow "Canlı demo" -Title "Sistem panoraması" -Subtitle "Güncel ve tarihsel kapsam, model hazırlığı ve veri yoğunluğu tek ekranda." -ImagePath (Join-Path $ScreenshotDirectory "01-genel-bakis.png") -CalloutTitle "Şeffaf sistem durumu" -CalloutBody "Bağlantı, GPU, model ve veri kapsamı kullanıcıdan saklanmaz. API kısmi çalışıyorsa yalnız ilgili görünüm etkilenir.")
    [void](Add-ScreenshotSlide -Presentation $presentation -Number 8 -Eyebrow "Canlı demo" -Title "Kaynaklı ürün karşılaştırması" -Subtitle "Ürün kodları güncel ve tarihsel uçlarda merkezi olarak eşlenir; sonuçlar kanıt düzeyinde ayrıştırılır." -ImagePath (Join-Path $ScreenshotDirectory "03-karsilastirma.png") -CalloutTitle "Koşul → belge" -CalloutBody "Her değer kendi belge ve kanıt cümlesiyle ilişkilidir. Eksik alan uydurulmaz; doğrulanmamış çıkarım açıkça işaretlenir.")
    [void](Add-ScreenshotSlide -Presentation $presentation -Number 9 -Eyebrow "Canlı demo" -Title "Kaynaklı yapay zekâ asistanı" -Subtitle "Hibrit aramayla bulunan resmî metinler, yerel Qwen yanıtında benzersiz dipnotlarla gösterilir." -ImagePath (Join-Path $ScreenshotDirectory "04-akilli-asistan.png") -CalloutTitle "RAG güvenlik sınırı" -CalloutBody "Yanıt finansal tavsiye değildir. Kaynak yoksa sistem bilgi uydurmaz; sohbet kapsamı güncel veya tarihsel seçilir.")

    $slide = Add-BaseSlide -Presentation $presentation
    Add-Header -Slide $slide -Eyebrow "Güvenlik ve kalite" -Title "Dış ağa açılmadan önce güvenli varsayılanlar" -Subtitle "Yönetim işlemleri public karar destek API'sinden ayrılır; veri statüsü sözleşmenin parçasıdır." -Number 10
    Add-FeatureCard -Slide $slide -Index "01" -Title "Public / admin ayrımı" -Body "Belge yazımı ve inceleme kararları API anahtarı ister; anahtar OpenAPI security scheme içinde tanımlıdır." -Left 48 -Top 155 -Width 410 -Height 145
    Add-FeatureCard -Slide $slide -Index "02" -Title "İstek korumaları" -Body "CORS allowlist, body-size sınırı ve hız sınırlama güvenli ortam değişkenleriyle yönetilir." -Left 478 -Top 155 -Width 434 -Height 145
    Add-FeatureCard -Slide $slide -Index "03" -Title "Veri bütünlüğü" -Body "Belge güncellemesi eski fact/entity artıklarını temizler; review kararı ve yazım atomik ve tek kayda bağlıdır." -Left 48 -Top 320 -Width 410 -Height 145
    Add-FeatureCard -Slide $slide -Index "04" -Title "Doğrulama görünürlüğü" -Body "İnsan doğrulaması tamamlanmamış her sonuç `model çıkarımı / doğrulanmadı` etiketi ve uyarı alanı taşır." -Left 478 -Top 320 -Width 434 -Height 145
    Add-Footer -Slide $slide

    $slide = Add-BaseSlide -Presentation $presentation
    Add-Header -Slide $slide -Eyebrow "Ölçüm" -Title "Başarı metrikleri ve veri kalite sınırları" -Subtitle "Performans değerleri silver/kural tabanlı etiketli test seti bağlamında raporlanır." -Number 11
    Add-MetricCard -Slide $slide -Value "0,9912" -Label "kampanya accuracy" -Left 48 -Top 170 -Width 195
    Add-MetricCard -Slide $slide -Value "0,9123" -Label "ürün accuracy" -Left 263 -Top 170 -Width 195
    Add-MetricCard -Slide $slide -Value "0,8685" -Label "NER entity F1" -Left 478 -Top 170 -Width 195
    Add-MetricCard -Slide $slide -Value "0,9828" -Label "NER token accuracy" -Left 693 -Top 170 -Width 219
    [void](Add-Rectangle -Slide $slide -Left 48 -Top 300 -Width 410 -Height 160 -Fill $Colors.Panel -Rounded -Line $Colors.Line)
    [void](Add-Text -Slide $slide -Text "Doğrulanan teknik kontroller" -Left 70 -Top 322 -Width 360 -Height 25 -Size 17 -Bold)
    [void](Add-Text -Slide $slide -Text "✓ JSONL bütünlüğü ve span ofsetleri`n✓ Train/val/test sızıntı denetimi`n✓ PostgreSQL dump parse kontrolü`n✓ API entegrasyon ve smoke testleri" -Left 70 -Top 360 -Width 350 -Height 82 -Size 13 -Color $Colors.Muted)
    [void](Add-Rectangle -Slide $slide -Left 478 -Top 300 -Width 434 -Height 160 -Fill "4A3B1A" -Rounded -Line $Colors.Amber)
    [void](Add-Text -Slide $slide -Text "Sınır ve sorumlu sunum" -Left 500 -Top 322 -Width 390 -Height 25 -Size 17 -Color $Colors.Amber -Bold)
    [void](Add-Text -Slide $slide -Text "Bu metrikler insan doğrulaması tamamlanmış gold veri anlamına gelmez. Finansal koşullar kullanıcıya resmî kaynak ve doğrulama statüsüyle sunulur." -Left 500 -Top 360 -Width 380 -Height 72 -Size 13 -Color $Colors.White)
    Add-Footer -Slide $slide

    $slide = Add-BaseSlide -Presentation $presentation
    Add-Header -Slide $slide -Eyebrow "Teslim ve işletim" -Title "Temiz klondan tekrarlanabilir kurulum" -Subtitle "Kod, veri, şema, model manifesti ve test hattı aynı sürüm sözleşmesiyle teslim edilir." -Number 12
    Add-FeatureCard -Slide $slide -Index "01" -Title "Kurulum sözleşmesi" -Body ".env.example, sürümlü SQL migration, Docker Compose ve least-privilege DB rolü." -Left 48 -Top 155 -Width 270 -Height 200
    Add-FeatureCard -Slide $slide -Index "02" -Title "Model paketi" -Body "Ağırlıklar sürümlü ZIP ve SHA-256 manifestiyle hazırdır; indirme URL'si yalnız başarılı release yüklemesinden sonra kaydedilir." -Left 345 -Top 155 -Width 270 -Height 200
    Add-FeatureCard -Slide $slide -Index "03" -Title "Otomatik kalite kapısı" -Body "Windows/Linux frontend lint, typecheck, test ve build; backend unit + migration smoke; bağımlılık audit." -Left 642 -Top 155 -Width 270 -Height 200
    [void](Add-Rectangle -Slide $slide -Left 48 -Top 390 -Width 864 -Height 80 -Fill $Colors.TealDark -Rounded)
    [void](Add-Text -Slide $slide -Text "Teslim referansı: BilisimVadisi2026 etiketi → doğrulanmış final commit" -Left 80 -Top 412 -Width 800 -Height 32 -Size 20 -Bold -Align 2)
    Add-Footer -Slide $slide

    $slide = Add-BaseSlide -Presentation $presentation
    [void](Add-Rectangle -Slide $slide -Left 0 -Top 0 -Width 960 -Height 540 -Fill $Colors.TealDark)
    [void](Add-Rectangle -Slide $slide -Left 80 -Top 70 -Width 800 -Height 400 -Fill $Colors.Navy -Rounded)
    [void](Add-Text -Slide $slide -Text "HITITFINLEX" -Left 130 -Top 110 -Width 700 -Height 24 -Size 13 -Color $Colors.Teal -Bold -Align 2)
    [void](Add-Text -Slide $slide -Text "Finansal bilgiye`nkaynağıyla güvenin." -Left 130 -Top 155 -Width 700 -Height 110 -Size 40 -Bold -Align 2)
    [void](Add-Text -Slide $slide -Text "Yerel · Kaynaklı · Tarihsel · Denetlenebilir" -Left 130 -Top 285 -Width 700 -Height 28 -Size 18 -Color $Colors.Cyan -Align 2)
    [void](Add-Text -Slide $slide -Text "github.com/abdulkadiripek/HititFinLex-Teknofest2026" -Left 130 -Top 340 -Width 700 -Height 24 -Size 14 -Color $Colors.White -Align 2)
    [void](Add-Text -Slide $slide -Text "Teşekkürler" -Left 130 -Top 395 -Width 700 -Height 28 -Size 17 -Color $Colors.Muted -Bold -Align 2)
    [void](Add-Text -Slide $slide -Text "13" -Left 887 -Top 500 -Width 30 -Height 18 -Size 10 -Color $Colors.Navy -Bold -Align 3)

    return $presentation
}

function New-DemoPresentation {
    param([Parameter(Mandatory)]$PowerPoint)
    $presentation = $PowerPoint.Presentations.Add()
    $presentation.PageSetup.SlideWidth = 960
    $presentation.PageSetup.SlideHeight = 540

    $slide = Add-BaseSlide -Presentation $presentation
    [void](Add-Text -Slide $slide -Text "HITITFINLEX" -Left 80 -Top 80 -Width 800 -Height 30 -Size 14 -Color $Colors.Teal -Bold -Align 2)
    [void](Add-Text -Slide $slide -Text "80 saniyede teknik demo" -Left 80 -Top 145 -Width 800 -Height 70 -Size 42 -Bold -Align 2)
    [void](Add-Text -Slide $slide -Text "Güncel + tarihsel veri · Kaynaklı karşılaştırma · Yerel RAG" -Left 100 -Top 240 -Width 760 -Height 35 -Size 19 -Color $Colors.Muted -Align 2)
    [void](Add-Text -Slide $slide -Text "#BilisimVadisi2026" -Left 100 -Top 345 -Width 760 -Height 30 -Size 15 -Color $Colors.Cyan -Bold -Align 2)
    Set-SlideTiming -Slide $slide -Seconds 8

    $screens = @(
        @{ Image = "01-genel-bakis.png"; Title = "1 · Sistem panoraması"; Body = "Canlı API, yerel GPU ve güncel/tarihsel veri kapsamı tek bakışta."; Seconds = 12 },
        @{ Image = "02-urun-katalogu.png"; Title = "2 · Akıllı katalog"; Body = "Banka, ürün, dönem, güven ve bilgi kapsamıyla 3.351 kaydı keşfedin."; Seconds = 12 },
        @{ Image = "03-karsilastirma.png"; Title = "3 · Kaynaklı karşılaştırma"; Body = "Koşulları banka bazında yan yana görün; her değer kendi kanıtına bağlıdır."; Seconds = 14 },
        @{ Image = "04-akilli-asistan.png"; Title = "4 · Yerel RAG asistanı"; Body = "BGE-M3 hibrit arama ve Qwen ile resmî kaynaklardan dipnotlu yanıt alın."; Seconds = 14 },
        @{ Image = "05-veri-kalitesi.png"; Title = "5 · Veri kalitesi"; Body = "Silver/model çıkarımı, güven ve insan inceleme durumu açıkça görünür."; Seconds = 12 }
    )

    foreach ($screen in $screens) {
        $slide = Add-BaseSlide -Presentation $presentation
        [void](Add-Text -Slide $slide -Text $screen.Title -Left 48 -Top 35 -Width 864 -Height 42 -Size 29 -Bold)
        [void](Add-Text -Slide $slide -Text $screen.Body -Left 48 -Top 82 -Width 864 -Height 30 -Size 14 -Color $Colors.Muted)
        [void](Add-Rectangle -Slide $slide -Left 48 -Top 130 -Width 864 -Height 356 -Fill $Colors.NavySoft -Rounded -Line $Colors.Line)
        $imagePath = Join-Path $ScreenshotDirectory $screen.Image
        if (Test-Path -LiteralPath $imagePath) {
            [void]$slide.Shapes.AddPicture($imagePath, 0, -1, 58, 140, 844, 336)
        }
        [void](Add-Text -Slide $slide -Text "Finansal tavsiye değildir · Sonuçlar resmî kaynakları ve doğrulama statüsünü içerir." -Left 48 -Top 508 -Width 864 -Height 18 -Size 9 -Color $Colors.Muted -Align 2)
        Set-SlideTiming -Slide $slide -Seconds $screen.Seconds
    }

    $slide = Add-BaseSlide -Presentation $presentation
    [void](Add-Text -Slide $slide -Text "Kaynağıyla karar desteği" -Left 80 -Top 155 -Width 800 -Height 60 -Size 40 -Bold -Align 2)
    [void](Add-Text -Slide $slide -Text "HititFinLex · Yerel · Kaynaklı · Denetlenebilir" -Left 80 -Top 245 -Width 800 -Height 35 -Size 19 -Color $Colors.Cyan -Align 2)
    [void](Add-Text -Slide $slide -Text "github.com/abdulkadiripek/HititFinLex-Teknofest2026" -Left 80 -Top 330 -Width 800 -Height 25 -Size 14 -Color $Colors.Muted -Align 2)
    Set-SlideTiming -Slide $slide -Seconds 8

    return $presentation
}

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

$PowerPoint = $null
$MainPresentation = $null
$DemoPresentation = $null
try {
    $PowerPoint = New-Object -ComObject PowerPoint.Application
    $PowerPoint.Visible = -1
    $PowerPoint.DisplayAlerts = 1

    $MainPresentation = New-MainPresentation -PowerPoint $PowerPoint
    $MainPresentation.SaveAs($PresentationPath, 24)
    $MainPresentation.Close()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($MainPresentation) | Out-Null
    $MainPresentation = $null
    Set-OpenXmlCoreMetadata -Path $PresentationPath -Title "HititFinLex TEKNOFEST 2026 Sunumu" -Subject "Yapay Zekâ Dil Ajanları Yarışması · 2. Senaryo"

    $MainPresentation = $PowerPoint.Presentations.Open($PresentationPath, 0, 0, 0)
    $MainPresentation.SaveAs($PdfPath, 32)
    $MainPresentation.Close()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($MainPresentation) | Out-Null
    $MainPresentation = $null

    $DemoPresentation = New-DemoPresentation -PowerPoint $PowerPoint
    $DemoPresentation.SaveAs($DemoSourcePath, 24)

    if (-not $SkipVideo) {
        $DemoPresentation.CreateVideo($DemoVideoPath, $true, 8, 720, 30, 85)
        $deadline = [DateTime]::UtcNow.AddMinutes(12)
        while ($DemoPresentation.CreateVideoStatus -in 1, 2) {
            if ([DateTime]::UtcNow -gt $deadline) {
                throw "Demo videosu 12 dakika içinde tamamlanamadı."
            }
            Start-Sleep -Seconds 2
        }
        if ($DemoPresentation.CreateVideoStatus -ne 3) {
            throw "PowerPoint demo videosunu dışa aktaramadı (durum: $($DemoPresentation.CreateVideoStatus))."
        }
    }

    $DemoPresentation.Close()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($DemoPresentation) | Out-Null
    $DemoPresentation = $null
    Set-OpenXmlCoreMetadata -Path $DemoSourcePath -Title "HititFinLex 80 Saniyelik Teknik Demo" -Subject "Kaynaklı katılım finansı karar desteği demosu"
} finally {
    if ($DemoPresentation) {
        $DemoPresentation.Close()
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($DemoPresentation) | Out-Null
    }
    if ($MainPresentation) {
        $MainPresentation.Close()
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($MainPresentation) | Out-Null
    }
    if ($PowerPoint) {
        $PowerPoint.Quit()
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($PowerPoint) | Out-Null
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}

$deliveryFiles = @(
    $PresentationPath,
    $PdfPath,
    $DemoSourcePath,
    $DemoVideoPath
) | Where-Object { Test-Path -LiteralPath $_ }
$checksumLines = $deliveryFiles | ForEach-Object {
    $hash = (Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash.ToLowerInvariant()
    "$hash  $([System.IO.Path]::GetFileName($_))"
}
[System.IO.File]::WriteAllLines(
    $ChecksumPath,
    $checksumLines,
    [System.Text.UTF8Encoding]::new($false)
)

Write-Output "PPTX: $PresentationPath"
Write-Output "PDF:  $PdfPath"
Write-Output "Demo: $DemoSourcePath"
if (-not $SkipVideo) { Write-Output "MP4:  $DemoVideoPath" }
Write-Output "SHA:  $ChecksumPath"
