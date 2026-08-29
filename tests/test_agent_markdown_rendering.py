"""Tests for offline Markdown and KaTeX rendering in RIXS Co-Pilot chat interface."""

from pathlib import Path
import subprocess
import pytest

from rixs_app.agent.system_prompt import build_system_prompt


def test_vendor_assets_exist():
    """Verify all necessary offline vendor assets (marked.js, KaTeX JS/CSS, fonts) are present."""
    vendor_dir = Path(__file__).parent.parent / 'rixs_app' / 'ui' / 'agent_sidebar' / 'templates' / 'vendor'
    
    assert (vendor_dir / 'marked.min.js').is_file(), 'marked.min.js missing'
    assert (vendor_dir / 'katex.min.js').is_file(), 'katex.min.js missing'
    assert (vendor_dir / 'katex.min.css').is_file(), 'katex.min.css missing'
    
    fonts_dir = vendor_dir / 'fonts'
    assert fonts_dir.is_dir(), 'fonts directory missing'
    woff2_files = list(fonts_dir.glob('*.woff2'))
    assert len(woff2_files) >= 15, f'Expected at least 15 woff2 font files, found {len(woff2_files)}'


def test_chat_html_template_integration():
    """Verify chat.html loads vendor assets and includes markdown & KaTeX styling."""
    template_path = Path(__file__).parent.parent / 'rixs_app' / 'ui' / 'agent_sidebar' / 'templates' / 'chat.html'
    content = template_path.read_text(encoding='utf-8')
    
    assert 'vendor/katex.min.css' in content
    assert 'vendor/marked.min.js' in content
    assert 'vendor/katex.min.js' in content
    assert 'parseMarkdown' in content
    assert 'katex.renderToString' in content
    assert '.katex-display' in content
    assert '.agent h1' in content
    assert '.agent table' in content


def test_system_prompt_includes_sidebar_formatting_directives():
    """Verify system prompt informs the agent about compact sidebar constraints and KaTeX syntax."""
    prompt = build_system_prompt(terminal_access=False)
    assert 'compact desktop sidebar widget' in prompt
    assert 'KaTeX' in prompt
    assert 'LaTeX' in prompt
    assert '###' in prompt


def test_js_markdown_katex_pipeline():
    """Test full parseMarkdown JS function in Node to ensure math formulas, headings, lists and code render."""
    template_dir = Path(__file__).parent.parent / 'rixs_app' / 'ui' / 'agent_sidebar' / 'templates'
    
    test_script = r"""
const marked = require('__TEMPLATE_DIR__/vendor/marked.min.js');
const katex = require('__TEMPLATE_DIR__/vendor/katex.min.js');

function parseMarkdown(text) {
    if (!text) return '';
    const codeTokens = [];
    let cleanText = text.replace(/(```[\s\S]*?```|`[^`\n]+`)/g, function(match) {
        const id = '%%CODE_TOKEN_' + codeTokens.length + '%%';
        codeTokens.push(match);
        return id;
    });

    const mathTokens = [];
    cleanText = cleanText.replace(/(\$\$([\s\S]*?)\$\$|\\\[([\s\S]*?)\\\])/g, function(match, p1, p2, p3) {
        const formula = p2 !== undefined ? p2 : p3;
        try {
            const html = katex.renderToString(formula.trim(), { displayMode: true, throwOnError: false });
            const id = '%%KATEX_TOKEN_' + mathTokens.length + '%%';
            mathTokens.push(html);
            return '\n\n' + id + '\n\n';
        } catch (e) {
            return match;
        }
    });

    cleanText = cleanText.replace(/(\$([^\$\n]+?)\$|\\\((.+?)\\\))/g, function(match, p1, p2, p3) {
        const formula = p2 !== undefined ? p2 : p3;
        try {
            const html = katex.renderToString(formula.trim(), { displayMode: false, throwOnError: false });
            const id = '%%KATEX_TOKEN_' + mathTokens.length + '%%';
            mathTokens.push(html);
            return id;
        } catch (e) {
            return match;
        }
    });

    for (let i = 0; i < codeTokens.length; i++) {
        cleanText = cleanText.replace('%%CODE_TOKEN_' + i + '%%', codeTokens[i]);
    }

    marked.setOptions({ breaks: true, gfm: true });
    let html = marked.parse(cleanText);

    for (let i = 0; i < mathTokens.length; i++) {
        html = html.split('%%KATEX_TOKEN_' + i + '%%').join(mathTokens[i]);
    }
    return html;
}

const sample = '### 1. Spatial Drift Alignment\n- **ECC**: Enhanced\n- Math: $R = E_{mono} / FWHM_{eV}$\n\n$$E = mc^2$$\n\nCode: `echo $VAR`';
const res = parseMarkdown(sample);
if (!res.includes('<h3')) process.exit(1);
if (!res.includes('class="katex"')) process.exit(2);
if (!res.includes('class="katex-display"')) process.exit(3);
if (!res.includes('<ul')) process.exit(4);
if (!res.includes('<code>echo $VAR</code>')) process.exit(5);
console.log('OK');
""".replace('__TEMPLATE_DIR__', str(template_dir))

    res = subprocess.run(['node', '-e', test_script], capture_output=True, text=True)
    assert res.returncode == 0, f'Node script failed: {res.stderr}'
    assert 'OK' in res.stdout
