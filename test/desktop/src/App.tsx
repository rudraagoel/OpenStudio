import { useState, useEffect } from 'react';
import { 
  Film, Sparkles, UserCheck, Mic, Upload, Image as ImageIcon,
  Zap, Play, AlertTriangle, MonitorPlay, MessageSquare,
  Scissors, AlignCenter, MonitorSmartphone, Cpu, 
  Activity, Database, Plus, X
} from 'lucide-react';
import './index.css';

const MODES = [
  { id: 'cinematic', icon: Film, name: 'Cinematic Videos', desc: 'Epic pacing, dramatic music, 4K render.' },
  { id: 'ad', icon: MonitorSmartphone, name: 'Advertisement', desc: 'Product ads with logos & overlays.' },
  { id: 'mixed', icon: Sparkles, name: 'Mixed Auto-Mode', desc: 'Auto-selects models & styles up to 10m.' },
  { id: 'character-story', icon: UserCheck, name: 'Character Story', desc: 'Narrative with consistent AI characters.' },
  { id: 'speaking', icon: Mic, name: 'Speaking Characters', desc: 'AI presenter with lip-sync.' },
  { id: 'faceless', icon: UserCheck, name: 'Faceless Story', desc: 'Cinematic narration over B-roll.' },
  { id: 'yapstyle', icon: MessageSquare, name: 'YapStyle', desc: 'Avatar clone from single reference.' },
  { id: 'caption', icon: AlignCenter, name: 'Captioning', desc: 'Hormozi/TikTok style text.' },
  { id: 'autoedit', icon: Scissors, name: 'Auto Edit', desc: 'AI edit raw folder footage.' },
  { id: 'image', icon: ImageIcon, name: 'Image Gen', desc: 'FLUX.1-dev generation.' }
];

interface ImageReference {
  path: string;
  subprompt: string;
}

export default function App() {
  const [tab, setTab] = useState('studio');
  
  // Execution State
  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [eta, setEta] = useState<string | null>(null);
  const [outputUrl, setOutputUrl] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [consoleLog, setConsoleLog] = useState<string[]>([]);

  // Studio Inputs
  const [mode, setMode] = useState('cinematic');
  const [prompt, setPrompt] = useState('A sleek futuristic hover-car flying through a neon-lit cyberpunk rain at night, ultra cinematic, 4K');
  const [images, setImages] = useState<ImageReference[]>([]);
  const [videoRef, setVideoRef] = useState<string | null>(null);
  const [folderRef, setFolderRef] = useState<string | null>(null);
  
  // Engine Controls
  const [modelId, setModelId] = useState('wan-t2v-1.3b');
  const [cfg, setCfg] = useState(6.0);
  const [offload, setOffload] = useState('balanced');
  
  // Voice & Video Studio Settings
  const [pitch, setPitch] = useState(0);
  const [speed, setSpeed] = useState(1.0);
  const [emotion, setEmotion] = useState('neutral');
  const [voice, setVoice] = useState('af_heart');
  const [isPreviewing, setIsPreviewing] = useState(false);
  
  // Telemetry
  const [vramUsed, setVramUsed] = useState(0);
  const [vramTotal, setVramTotal] = useState(4);
  const [gpuName, setGpuName] = useState('NVIDIA RTX (Detecting...)');

  useEffect(() => {
    // Hardware Telemetry Polling
    const interval = setInterval(async () => {
      try {
        // @ts-ignore
        if (window.opencanon && window.opencanon.getTelemetry) {
          // @ts-ignore
          const stats = await window.opencanon.getTelemetry();
          if (stats.vramTotal > 0) {
            setVramUsed(stats.vramUsed);
            setVramTotal(stats.vramTotal);
            setGpuName(stats.gpuName);
          }
        }
      } catch (e) {
        // ignore telemetry errors to prevent UI crashes
      }
    }, 2000);

    // IPC Command Listener
    // @ts-ignore
    if (window.opencanon && window.opencanon.onCommandOutput) {
      // @ts-ignore
      window.opencanon.onCommandOutput((msg: string) => {
        setConsoleLog(prev => [...prev, msg].slice(-20)); // Keep last 20 lines
        const percentMatch = msg.match(/(\d{1,3})%\|/);
        if (percentMatch) setProgress(parseInt(percentMatch[1], 10));
        
        const etaMatch = msg.match(/<(\d{2}:\d{2})/);
        if (etaMatch) setEta(etaMatch[1]);
        
        const savedMatch = msg.match(/Saved (?:video|clip|reel|image) to:\s*([^\n]+)/i);
        if (savedMatch) setOutputUrl(`file:///${savedMatch[1].trim().replace(/\\/g, '/')}`);
      });
    }

    return () => clearInterval(interval);
  }, [isGenerating]);

  const handlePreviewVoice = async () => {
    setIsPreviewing(true);
    try {
      const outputPath = '../desktop/public/preview.wav';
      const cmd = `python -m cli.main generate tts --text "This is a preview of the selected voice." --voice ${voice} --output ${outputPath}`;
      // @ts-ignore
      await window.opencanon.runCommand(cmd);
      const audio = new Audio('/preview.wav?t=' + Date.now());
      audio.play();
    } catch (err) {
      console.error(err);
    }
    setIsPreviewing(false);
  };

  const executeCLI = async (cmd: string) => {
    setIsGenerating(true); setProgress(0); setEta(null); setOutputUrl(null); setErrorMsg(null); setConsoleLog([]);
    try {
      // Local GPU Execution
      // @ts-ignore
      const result = await window.opencanon.runCommand(cmd);
      if (result && result.includes('Process exited with code 1')) {
        setErrorMsg('Engine crashed: ' + result.slice(0, 150) + '...');
      } else {
        setProgress(100);
        
        // --- EMBEDDED REMOTION PIPELINE ---
        const savedMatch = result.match(/Saved (?:video|clip|reel|image) to:\s*([^\n]+)/i);
        if (savedMatch && (mode === 'ad' || mode === 'caption')) {
          const videoPath = savedMatch[1].trim().replace(/\\/g, '/');
          setConsoleLog(prev => [...prev, "[SYSTEM] Starting Remotion VFX Compositing Pass..."]);
          
          let propsObj: any = { backgroundVideoUrl: `file:///${videoPath}` };
          if (mode === 'caption') propsObj.captions = prompt.split(" ").filter(w => w.length > 2);
          if (mode === 'ad' && images.length > 0 && images[0].path) propsObj.logoUrl = `file:///${images[0].path.replace(/\\/g, '/')}`;
          
          // Use powershell to write props.json to avoid Windows quoting nightmares with npx
          const remotionCmd = `Set-Content -Path '../remotion/props.json' -Value '${JSON.stringify(propsObj)}' -Encoding utf8; cd ../remotion ; npx remotion render src/index.ts AIComposition out.mp4 --props props.json`;
          // @ts-ignore
          await window.opencanon.runCommand(remotionCmd);
          setConsoleLog(prev => [...prev, "[SYSTEM] Remotion VFX complete. Updating output preview."]);
          
          // Hardcode remotion output path relative to execution
          setOutputUrl(`file:///C:/OpenCanon AI Studio/test/remotion/out.mp4`);
        }
      }
    } catch (err: any) { setErrorMsg(err.toString()); }
    setIsGenerating(false);
  };

  const handleGenerate = () => {
    let cmd = '';
    
    // Auto-fetch length from prompt (e.g. "4 seconds", "10 minutes", "5m", "15s")
    let autoLength = '1m'; // default to 1 minute
    const timeRegex = /(\d+)\s*(s|sec|seconds|m|min|minutes)/i;
    const match = prompt.match(timeRegex);
    if (match) {
      const val = parseInt(match[1]);
      const unit = match[2].toLowerCase();
      autoLength = unit.startsWith('s') ? `${val}s` : `${val}m`;
    }

    const baseArgs = `--quality ${offload} --model ${modelId} --duration ${autoLength}`;
    const voiceArgs = `--pitch ${pitch} --emotion ${emotion} --speed ${speed} --voice ${voice}`;
    
    // Inject multi-image prompt syntax natively supported by CLI
    let finalPrompt = prompt;
    const validImages = images.filter(img => img.path && img.path !== 'undefined');
    if (validImages.length > 0) {
      finalPrompt += " Reference Context: ";
      validImages.forEach(img => {
        finalPrompt += `${img.subprompt} @${img.path} `;
      });
    }
    
    // Strip double quotes and newlines to prevent PowerShell syntax parsing crashes on external commands
    finalPrompt = finalPrompt.replace(/"/g, '').replace(/\n/g, ' ');
    
    switch (mode) {
      case 'cinematic':
        cmd = `python -m cli.main generate video --prompt '${finalPrompt.replace(/'/g, "''")}' --render-style cinematic ${baseArgs}`;
        if (validImages.length > 0) cmd += ` --image "${validImages[0].path}"`;
        break;
      case 'mixed':
        cmd = `python -m cli.main generate mixed --prompt '${finalPrompt.replace(/'/g, "''")}' ${baseArgs}`;
        if (validImages.length > 0) cmd += ` --image "${validImages[0].path}"`;
        break;
      case 'ad':
        cmd = `python -m cli.main generate video --prompt '${finalPrompt.replace(/'/g, "''")}' --render-style motion_graphics ${baseArgs}`;
        if (validImages.length > 0) cmd += ` --logo "${validImages[0].path}"`;
        break;
      case 'character-story':
        cmd = `python -m cli.main generate video --prompt '${finalPrompt.replace(/'/g, "''")}' --render-style cinematic --tts true ${baseArgs} ${voiceArgs}`;
        if (validImages.length > 0) cmd += ` --image "${validImages[0].path}"`;
        break;
      case 'speaking':
        cmd = `python -m cli.main character speak --character "Default" --text '${finalPrompt.replace(/'/g, "''")}' ${voiceArgs}`;
        break;
      case 'faceless':
        cmd = `python -m cli.main generate story --script '${finalPrompt.replace(/'/g, "''")}' --duration ${autoLength} ${voiceArgs}`;
        break;
      case 'yapstyle':
        cmd = `python -m cli.main generate yap --prompt '${finalPrompt.replace(/'/g, "''")}'`;
        if (validImages.length > 0) cmd += ` --image "${validImages[0].path}"`;
        break;
      case 'pixar':
        cmd = `python -m cli.main generate video --prompt '${finalPrompt.replace(/'/g, "''")}' --render-style 3d_pixar ${baseArgs}`;
        break;
      case 'caption':
        cmd = `python -m cli.main edit captions --video "${videoRef}"`;
        break;
      case 'autoedit':
        cmd = `python -m cli.main edit auto --folder "${folderRef}" --instructions '${finalPrompt.replace(/'/g, "''")}'`;
        break;
      case 'vfx':
        cmd = `python -m cli.main generate video --prompt '${finalPrompt.replace(/'/g, "''")}' --render-style motion_graphics ${baseArgs}`;
        break;
      case 'image':
        cmd = `python -m cli.main generate image --prompt '${finalPrompt.replace(/'/g, "''")}' --model flux.1-dev`;
        break;
    }
    if (cmd) executeCLI(cmd);
  };

  const handleTrain = (type: string) => {
    if (!folderRef) return setErrorMsg('Must provide a dataset folder.');
    const cmd = `python -m cli.main train ${type} --dataset "${folderRef}" --steps 500 --rank 16`;
    executeCLI(cmd);
  };

  const handleAddImage = async () => {
    // @ts-ignore
    const filePath = await window.opencanon.selectFile();
    if (filePath) {
      setImages([...images, { path: filePath, subprompt: '' }]);
    }
  };

  const updateImageSubprompt = (index: number, subprompt: string) => {
    const newImages = [...images];
    newImages[index].subprompt = subprompt;
    setImages(newImages);
  };

  const removeImage = (index: number) => {
    setImages(images.filter((_, i) => i !== index));
  };

  return (
    <div style={{ display: 'flex', height: '100vh', width: '100vw', position: 'relative' }}>
      <div className="bg-orb" style={{ width: '800px', height: '800px', background: 'var(--accent-purple)', top: '-200px', left: '-200px' }} />
      <div className="bg-orb" style={{ width: '600px', height: '600px', background: 'var(--accent-cyan)', bottom: '-100px', right: '-100px' }} />

      {/* Sidebar */}
      <div style={{ width: '80px', display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '24px 0', background: 'rgba(10, 10, 15, 0.8)', borderRight: '1px solid var(--panel-border)', zIndex: 10 }}>
        <Sparkles size={28} color="white" style={{ marginBottom: '40px' }} />
        {[
          { id: 'studio', icon: Film, title: 'Studio' },
          { id: 'train', icon: Database, title: 'Training Hub' },
          { id: 'telemetry', icon: Activity, title: 'Telemetry' }
        ].map(n => (
          <div key={n.id} onClick={() => setTab(n.id)} title={n.title}
               style={{ width: '48px', height: '48px', borderRadius: '14px', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '12px', cursor: 'pointer', transition: 'all 0.2s',
               background: tab === n.id ? 'linear-gradient(135deg, rgba(139, 92, 246, 0.2), rgba(6, 182, 212, 0.2))' : 'transparent',
               color: tab === n.id ? 'var(--accent-cyan)' : 'var(--text-muted)' }}>
            <n.icon size={20} />
          </div>
        ))}
      </div>

      {/* Main Container */}
      <div style={{ flex: 1, display: 'flex', padding: '24px', gap: '24px', overflow: 'hidden' }}>
        
        {/* LEFT PANEL */}
        <div style={{ flex: '0 0 500px', display: 'flex', flexDirection: 'column', gap: '20px', overflowY: 'auto', paddingRight: '10px' }}>
          
          <div className="glass-panel" style={{ padding: '24px' }}>
            {tab === 'studio' && (
              <>
                <h2 className="heading-font" style={{ fontSize: '24px', fontWeight: '800', marginBottom: '16px' }}>Creation Studio</h2>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '8px', marginBottom: '24px' }}>
                  {MODES.map(m => (
                    <div key={m.id} onClick={() => setMode(m.id)}
                      style={{ padding: '12px', borderRadius: '10px', cursor: 'pointer', border: `1px solid ${mode === m.id ? 'rgba(139,92,246,0.4)' : 'rgba(255,255,255,0.05)'}`, background: mode === m.id ? 'rgba(139,92,246,0.1)' : 'rgba(255,255,255,0.02)' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px', color: mode === m.id ? 'white' : 'var(--text-muted)' }}>
                        <m.icon size={14} /> <span style={{ fontWeight: '600', fontSize: '12px' }}>{m.name}</span>
                      </div>
                    </div>
                  ))}
                </div>

                <div style={{ borderTop: '1px solid var(--panel-border)', paddingTop: '20px' }}>
                  <label className="input-label" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}><Cpu size={14} /> Inference Engine Config</label>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginBottom: '12px' }}>
                    <select value={modelId} onChange={(e) => setModelId(e.target.value)} className="input-field">
                      <option value="wan-t2v-1.3b">Wan2.1-T2V (1.3B)</option>
                      <option value="ltx-video">LTX-Video 2.3</option>
                      <option value="flux.1-dev">FLUX.1-dev</option>
                    </select>
                    <select value={offload} onChange={(e) => setOffload(e.target.value)} className="input-field">
                      <option value="balanced">Balanced Offload</option>
                      <option value="fast">CPU RAM Fast</option>
                      <option value="ultra">Max VRAM Lock</option>
                      <option value="extreme_4gb">Extreme 4GB VRAM Lock</option>
                    </select>
                  </div>
                  <div style={{ display: 'flex', gap: '10px' }}>
                    <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>CFG: {cfg}</span>
                      <input type="range" min="1" max="15" step="0.5" value={cfg} onChange={(e) => setCfg(parseFloat(e.target.value))} style={{ width: '100%' }} />
                    </div>
                  </div>
                </div>

                <div style={{ borderTop: '1px solid var(--panel-border)', paddingTop: '20px', marginTop: '20px' }}>
                  <label className="input-label" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}><Mic size={14} /> Voice Studio Options</label>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '10px', marginBottom: '12px' }}>
                    <div style={{ display: 'flex', gap: '5px' }}>
                      <select value={voice} onChange={(e) => setVoice(e.target.value)} className="input-field" style={{ flex: 1 }}>
                        <option value="af_heart">Voice: Heart (F)</option>
                        <option value="af_alloy">Voice: Alloy (F)</option>
                        <option value="am_echo">Voice: Echo (M)</option>
                        <option value="am_michael">Voice: Michael (M)</option>
                      </select>
                      <button className="btn-primary" style={{ padding: '0 12px', whiteSpace: 'nowrap', borderRadius: '8px', cursor: 'pointer' }} onClick={handlePreviewVoice} disabled={isPreviewing}>
                        {isPreviewing ? '...' : '▶'}
                      </button>
                    </div>
                    <select value={emotion} onChange={(e) => setEmotion(e.target.value)} className="input-field">
                      <option value="neutral">Emotion: Neutral</option>
                      <option value="happy">Emotion: Happy</option>
                      <option value="dramatic">Emotion: Dramatic</option>
                      <option value="excited">Emotion: Excited</option>
                    </select>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '0 10px' }}>
                      <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Speed: {speed.toFixed(1)}x</span>
                      <input type="range" min="0.5" max="2.0" step="0.1" value={speed} onChange={(e) => setSpeed(parseFloat(e.target.value))} style={{ width: '100%' }} />
                    </div>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '0 10px' }}>
                    <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Pitch: {pitch > 0 ? `+${pitch}` : pitch}</span>
                    <input type="range" min="-3" max="3" step="0.1" value={pitch} onChange={(e) => setPitch(parseFloat(e.target.value))} style={{ width: '100%' }} />
                  </div>
                </div>
              </>
            )}

            {tab === 'train' && (
              <>
                <h2 className="heading-font" style={{ fontSize: '24px', fontWeight: '800', marginBottom: '16px' }}>LoRA Training Hub</h2>
                <p style={{ fontSize: '13px', color: 'var(--text-muted)', marginBottom: '20px' }}>Train custom stylistic or subject embeddings directly on your local GPU.</p>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  <label className="btn-ghost" style={{ cursor: 'pointer', flex: 1, justifyContent: 'center' }}>
                    {/* @ts-ignore */}
                    <input type="file" style={{ display: 'none' }} webkitdirectory="" directory="" onChange={(e: any) => { if (e.target.files && e.target.files[0]) setFolderRef(e.target.files[0].path); }} />
                    <Upload size={16} /> {folderRef ? 'Dataset Attached' : 'Select Dataset Folder'}
                  </label>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '8px' }}>
                    <button className="btn-ghost" onClick={() => handleTrain('video')}>Train Video LoRA</button>
                    <button className="btn-ghost" onClick={() => handleTrain('image')}>Train Image LoRA</button>
                    <button className="btn-ghost" onClick={() => handleTrain('voice')}>Train Voice</button>
                  </div>
                </div>
              </>
            )}

            {tab === 'telemetry' && (
              <>
                <h2 className="heading-font" style={{ fontSize: '24px', fontWeight: '800', marginBottom: '16px' }}>Hardware Telemetry</h2>
                <div style={{ background: 'rgba(0,0,0,0.5)', padding: '16px', borderRadius: '12px', border: '1px solid var(--panel-border)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                    <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>GPU VRAM ({gpuName})</span>
                    <span style={{ fontSize: '12px', color: 'var(--accent-pink)', fontWeight: 'bold' }}>{vramUsed.toFixed(1)} / {vramTotal.toFixed(1)} GB</span>
                  </div>
                  <div className="progress-container"><div className="progress-fill" style={{ width: `${Math.min((vramUsed/vramTotal)*100, 100)}%`, background: 'var(--accent-pink)' }} /></div>
                  <p style={{ marginTop: '16px', fontSize: '12px', color: 'var(--text-muted)' }}>Supabase P2P Hub is currently disabled per local configuration.</p>
                </div>
              </>
            )}
          </div>
        </div>

        {/* RIGHT PANEL: Execution */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '20px' }}>
          
          {(tab === 'studio' || tab === 'train') && (
            <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column' }}>
              <label className="input-label">Prompt / Script / Instructions</label>
              <textarea rows={4} value={prompt} onChange={(e) => setPrompt(e.target.value)} className="input-field" style={{ marginBottom: '16px' }} />
              
              {/* Multi-Image Subprompt Section */}
              <div style={{ marginBottom: '16px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                  <label className="input-label" style={{ marginBottom: 0 }}>Reference Images & Subprompts</label>
                  <button onClick={handleAddImage} className="btn-ghost" style={{ cursor: 'pointer', padding: '6px 12px', fontSize: '11px', background: 'transparent', border: '1px solid rgba(255,255,255,0.1)' }}>
                    <Plus size={14} /> Add Image
                  </button>
                </div>
                
                {images.length > 0 ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', maxHeight: '150px', overflowY: 'auto' }}>
                    {images.map((img, i) => (
                      <div key={i} style={{ display: 'flex', gap: '8px', alignItems: 'center', background: 'rgba(255,255,255,0.03)', padding: '8px', borderRadius: '8px' }}>
                        <div style={{ width: '40px', height: '40px', background: 'rgba(0,0,0,0.3)', borderRadius: '6px', display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
                          {img.path ? <img src={`file:///${img.path.replace(/\\/g, '/')}`} style={{width: '100%', height: '100%', objectFit: 'cover'}} alt="ref" /> : <ImageIcon size={16} color="var(--text-muted)" />}
                        </div>
                        <input type="text" className="input-field" style={{ flex: 1, padding: '8px 12px' }} placeholder="What is this image? How should it be used?" value={img.subprompt} onChange={(e) => updateImageSubprompt(i, e.target.value)} />
                        <button className="btn-ghost" style={{ padding: '8px' }} onClick={() => removeImage(i)}><X size={16} color="#ef4444" /></button>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div style={{ fontSize: '12px', color: 'var(--text-muted)', fontStyle: 'italic', padding: '8px 0' }}>No reference images attached.</div>
                )}
              </div>

              <div style={{ display: 'flex', gap: '12px', marginBottom: '16px' }}>
                {mode === 'caption' && (
                  <label className="btn-ghost" style={{ cursor: 'pointer', flex: 1, justifyContent: 'center' }}>
                    <input type="file" style={{ display: 'none' }} accept="video/*" onChange={(e: any) => { if (e.target.files && e.target.files[0]) setVideoRef(e.target.files[0].path); }} />
                    <Film size={16} /> {videoRef ? 'Video Attached' : 'Attach Video (Captions)'}
                  </label>
                )}
                {mode === 'autoedit' && (
                  <label className="btn-ghost" style={{ cursor: 'pointer', flex: 1, justifyContent: 'center' }}>
                    {/* @ts-ignore */}
                    <input type="file" style={{ display: 'none' }} webkitdirectory="" directory="" onChange={(e: any) => { if (e.target.files && e.target.files[0]) setFolderRef(e.target.files[0].path); }} />
                    <Upload size={16} /> {folderRef ? 'Folder Attached' : 'Attach Folder (Auto Edit)'}
                  </label>
                )}
              </div>

              <button className="btn-primary" onClick={handleGenerate} disabled={isGenerating}>
                <Zap size={18} /> {isGenerating ? 'Computing Tensors...' : 'Execute Generation Pipeline'}
              </button>
            </div>
          )}

          <div className="glass-panel" style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <div style={{ padding: '16px 24px', borderBottom: '1px solid var(--panel-border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span className="heading-font" style={{ fontWeight: '700', fontSize: '14px', letterSpacing: '0.05em', color: 'var(--text-muted)' }}>OUTPUT VIEWER</span>
              <span style={{ fontSize: '12px', color: isGenerating ? 'var(--accent-cyan)' : 'var(--text-muted)' }}>{isGenerating ? 'GPU ALLOCATED' : 'IDLE'}</span>
            </div>
            
            <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(0,0,0,0.6)', position: 'relative' }}>
              {errorMsg ? (
                <div style={{ padding: '24px', color: '#ef4444', textAlign: 'center' }}>
                  <AlertTriangle size={48} style={{ margin: '0 auto 16px' }} />
                  <p style={{ fontFamily: 'monospace', background: 'rgba(0,0,0,0.5)', padding: '16px', borderRadius: '8px' }}>{errorMsg}</p>
                </div>
              ) : outputUrl ? (
                outputUrl.endsWith('.png') || outputUrl.endsWith('.jpg') 
                  ? <img src={outputUrl} style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                  : <video src={outputUrl} autoPlay loop controls style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
              ) : isGenerating ? (
                <div style={{ width: '60%', textAlign: 'center' }}>
                  <Sparkles size={32} color="var(--accent-purple)" style={{ animation: 'float 2s infinite ease-in-out', margin: '0 auto 16px' }} />
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '8px' }}>
                    <h3 className="heading-font" style={{ fontSize: '24px', fontWeight: '800', margin: 0 }}>Processing Inference...</h3>
                    {eta && <span style={{ fontSize: '14px', color: 'var(--accent-purple)', fontWeight: 'bold' }}>ETA: {eta}</span>}
                  </div>
                  <div className="progress-container"><div className="progress-fill" style={{ width: `${progress}%` }} /></div>
                  <div style={{ marginTop: '24px', textAlign: 'left', fontFamily: 'monospace', fontSize: '10px', color: 'var(--text-muted)', background: 'rgba(0,0,0,0.4)', padding: '8px', borderRadius: '4px', height: '100px', overflowY: 'auto' }}>
                    {consoleLog.map((l, i) => <div key={i}>{l}</div>)}
                  </div>
                </div>
              ) : (
                <Play size={64} style={{ opacity: 0.3, color: 'white' }} />
              )}
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
