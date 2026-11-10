using System.Diagnostics;
using System.Text;
using System.Text.Json;
using Terminal.Gui.App;
using Terminal.Gui.Views;
using Terminal.Gui.ViewBase;
using Terminal.Gui.Drawing;
using Terminal.Gui.Input;
using Terminal.Gui.Configuration;
using Attribute = Terminal.Gui.Drawing.Attribute;

Console.InputEncoding = Encoding.UTF8;
Console.OutputEncoding = Encoding.UTF8;
string Arg(string name, string fallback) {
    int i = Array.IndexOf(args, name);
    return i >= 0 && i+1 < args.Length ? args[i+1] : fallback;
}
if (args.Contains("--self-test")) {
    if (JsonSerializer.Serialize(new {command="/help"}) != "{\"command\":\"/help\"}") return 2;
    Console.WriteLine("Shotsmith / Terminal.Gui 2.5.0 / protocol OK"); return 0;
}
var repo = Arg("--repo", Directory.GetCurrentDirectory());
var python = Arg("--python", "python");
using IApplication app = Application.Create().Init();
if(app.Driver is not null) app.Driver.Force16Colors=false;
if(args.Contains("--ui-smoke-wide")) app.Driver!.SetScreenSize(120,40);
using var window = new ShotsmithWindow(repo, python);
if(args.Contains("--ui-smoke-help")) {
    _=Task.Run(async()=>{await Task.Delay(800);app.Invoke(window.SmokeHelp);});
}
if(args.Contains("--ui-smoke-progress")) {
    _=Task.Run(async()=>{await Task.Delay(800);app.Invoke(window.SmokeProgress);});
}
if(args.Contains("--ui-smoke") || args.Contains("--ui-smoke-wide") || args.Contains("--ui-smoke-help") || args.Contains("--ui-smoke-progress")) {
    _=Task.Run(async()=> {await Task.Delay(4500);app.Invoke(()=>{
        File.WriteAllText(Arg("--capture",Path.Combine(repo,"shotsmith-screen.txt")),app.ToString());
        app.RequestStop();
    });});
}
app.Run(window);
return 0;

sealed class ShotsmithWindow : Runnable {
    readonly Label logo, help, context, footer, status, prompt;
    readonly View inputBox;
    readonly TextField input;
    readonly TextView log;
    readonly ProgressBar progress;
    readonly List<string> history = new();
    readonly StringBuilder transcript = new();
    Process? backend;
    bool busy, waiting;
    int historyIndex;
    readonly string repo, python;
    static readonly Color Background = new(25,26,35);
    static readonly Color Panel = new(35,37,53);
    static readonly Color Accent = new(153,168,245);
    static readonly Color Muted = new(130,137,171);
    static readonly Color Foreground = new(211,217,242);
    static Scheme Palette(Color fg, Color bg) => new() {
        Normal=new Attribute(fg,bg), Focus=new Attribute(fg,bg),
        HotNormal=new Attribute(Accent,bg), HotFocus=new Attribute(Accent,bg),
        Disabled=new Attribute(Muted,bg)
    };
    static string Register(Color fg, Color bg) { string key=fg.ToString()+bg.ToString(); SchemeManager.AddScheme(key,Palette(fg,bg)); return key; }
    const string Brand = " ███████╗██╗  ██╗ ██████╗ ████████╗███████╗███╗   ███╗██╗████████╗██╗  ██╗\n"+
                         " ██╔════╝██║  ██║██╔═══██╗╚══██╔══╝██╔════╝████╗ ████║██║╚══██╔══╝██║  ██║\n"+
                         " ███████╗███████║██║   ██║   ██║   ███████╗██╔████╔██║██║   ██║   ███████║\n"+
                         " ╚════██║██╔══██║██║   ██║   ██║   ╚════██║██║╚██╔╝██║██║   ██║   ██╔══██║\n"+
                         " ███████║██║  ██║╚██████╔╝   ██║   ███████║██║ ╚═╝ ██║██║   ██║   ██║  ██║\n"+
                         " ╚══════╝╚═╝  ╚═╝ ╚═════╝    ╚═╝   ╚══════╝╚═╝     ╚═╝╚═╝   ╚═╝   ╚═╝  ╚═╝";

    public ShotsmithWindow(string repo, string python) {
        this.repo=repo; this.python=python;
        Title="Shotsmith"; SchemeName=Register(Foreground,Background);
        CanFocus=true; Width=Dim.Fill(); Height=Dim.Fill();
        context=new Label {Text="SHOTSMITH  /  ваша студия сцен", X=2,Y=1,SchemeName=Register(Muted,Background)};
        logo=new Label {Text=Brand, X=Pos.Center(),Y=Pos.Percent(24),SchemeName=Register(Accent,Background)};
        help=new Label {X=Pos.Center(),Y=Pos.Bottom(logo)+2,Text=
            "/connect      подключить GPU\n/connections  сохранённые подключения\n/profile      bundle · qwen\n/install      подготовить установку\n/new          проект и референсы\n/frames       опорные кадры\n/video        генерация сцен\n/assemble     сборка ролика\n/help         все команды",SchemeName=Register(Muted,Background)};
        log=new TextView {X=3,Y=3,Width=Dim.Fill(3),Height=Dim.Fill(9),ReadOnly=true,Visible=false,SchemeName=Register(Foreground,Background),CanFocus=false};
        progress=new ProgressBar {X=Pos.Percent(10),Y=Pos.AnchorEnd(8),Width=Dim.Percent(80),Height=1,Visible=false,SchemeName=Register(Accent,Panel)};
        status=new Label {X=Pos.Percent(10),Y=Pos.AnchorEnd(7),Width=Dim.Percent(80),Height=1,Text="",SchemeName=Register(Muted,Background)};
        inputBox=new View {CanFocus=true,X=Pos.Percent(10),Y=Pos.AnchorEnd(5),Width=Dim.Percent(80),Height=3,SchemeName=Register(Foreground,Panel)};
        prompt=new Label {Text="›",X=1,Y=1,SchemeName=Register(Accent,Panel)};
        input=new TextField {CanFocus=true,X=3,Y=1,Width=Dim.Fill(2),Height=1,SchemeName=Register(Foreground,Panel)};
        inputBox.Add(prompt,input);
        footer=new Label {X=Pos.Percent(10),Y=Pos.AnchorEnd(2),Width=Dim.Percent(80),Height=1,Text="Enter отправить   Tab команды   ↑↓ история   Ctrl+Q выход",SchemeName=Register(Muted,Background)};
        Add(context,logo,help,log,progress,status,inputBox,footer);
        SetScheme(Palette(Foreground,Background));
        foreach(var v in new View[]{context,help,footer,status})v.SetScheme(Palette(Muted,Background));
        logo.SetScheme(Palette(Accent,Background));
        log.SetScheme(Palette(Foreground,Background));
        inputBox.SetScheme(Palette(Foreground,Panel));
        input.SetScheme(Palette(Foreground,Panel));
        prompt.SetScheme(Palette(Accent,Panel));
        progress.SetScheme(Palette(Accent,Panel));
        ViewportChanged += (_,_)=> {
            var small=(App?.Screen.Width??100)<90 || (App?.Screen.Height??40)<30;
            logo.Text=small?"S H O T S M I T H":Brand;
            logo.Height=small?1:6;
            help.Text=small?"/connect  подключить GPU\n/install  установка моделей\n/status   состояние\n/help     все команды":
                "/connect      подключить GPU\n/connections  сохранённые подключения\n/profile      bundle · qwen\n/install      подготовить установку\n/new          проект и референсы\n/frames       опорные кадры\n/video        генерация сцен\n/assemble     сборка ролика\n/help         все команды";
        };
        input.KeyDown += (_,e) => {
            if (e==Key.Enter) {Submit();e.Handled=true;}
            else if(e==Key.CursorUp && history.Count>0 && !waiting) {historyIndex=Math.Max(0,historyIndex-1);input.Text=history[historyIndex];e.Handled=true;}
            else if(e==Key.CursorDown && history.Count>0 && !waiting) {historyIndex=Math.Min(history.Count,historyIndex+1);input.Text=historyIndex==history.Count?"":history[historyIndex];e.Handled=true;}
            else if(e==Key.Tab && !waiting) {
                string[] commands={"/connect","/connections","/use","/profile","/inspect","/install","/status","/resume","/open","/disconnect","/new","/projects","/edit","/prepare","/frames","/approve","/storyboard","/video","/assemble","/outputs","/doctor","/project","/keyframes","/help","/exit"};
                var prefix=input.Text.ToString()??"";var matches=commands.Where(c=>c.StartsWith(prefix,StringComparison.OrdinalIgnoreCase)).ToArray();
                if(matches.Length==1)input.Text=matches[0]+" ";else status.Text=string.Join("  ",matches);
                e.Handled=true;
            }
        };
        KeyDown += (_,e)=> {if(e==Key.Esc&&!busy){logo.Visible=true;help.Visible=true;log.Visible=false;input.Text="";input.SetFocus();e.Handled=true;} if(e==Key.Q.WithCtrl){Exit();e.Handled=true;}};
        Initialized += (_,_)=> {input.SetFocus();};
        IsRunningChanging += (_,e)=> {
            if(!e.NewValue && busy){e.Cancel=true;status.Text="Команда выполняется · дождитесь завершения";}
        };
    }
    internal void SmokeHelp(){input.Text="/help";Submit();}
    internal void SmokeProgress(){Receive("{\"type\":\"progress\",\"text\":\"Демо шкалы · локальный тест · 39.3 / 61.2 GiB · ETA 35m\",\"ratio\":0.64}");}
    void ShowLog(string text) {
        logo.Visible=false;help.Visible=false;log.Visible=true;
        transcript.AppendLine(text);
        if(transcript.Length>80000)transcript.Remove(0,transcript.Length-60000);
        log.Text=transcript.ToString();
        log.MoveEnd();
    }
    void Submit() {
        var text=input.Text.ToString()??"";
        if(busy&&!waiting)return;
        if(!waiting&&string.IsNullOrWhiteSpace(text))return;
        if(!waiting&&(text.Trim()=="/exit"||text.Trim()=="/quit")){Exit();return;}
        input.Text="";
        if(waiting){waiting=false;status.Text="Выполняется…";Send(new {input=text});return;}
        if(!text.StartsWith('/')){status.Text="Команды начинаются с / · Tab — подсказки";return;}
        history.Add(text);historyIndex=history.Count;ShowLog("› "+text);
        if(!EnsureBackend())return;
        busy=true;status.Text="Выполняется…";
        Send(new {command=text});
    }
    bool EnsureBackend() {
        if(backend is {HasExited:false})return true;
        try {
            var info=new ProcessStartInfo(python) {WorkingDirectory=repo,UseShellExecute=false,CreateNoWindow=true,RedirectStandardInput=true,RedirectStandardOutput=true,RedirectStandardError=true,StandardOutputEncoding=Encoding.UTF8,StandardErrorEncoding=Encoding.UTF8,StandardInputEncoding=new UTF8Encoding(false)};
            info.ArgumentList.Add("-u");info.ArgumentList.Add("-B");info.ArgumentList.Add("-m");info.ArgumentList.Add("scripts.console_bridge");
            info.Environment["PYTHONIOENCODING"]="utf-8";
            backend=Process.Start(info)!;
            var process=backend;
            _=Task.Run(async ()=> {string? line;while((line=await process.StandardOutput.ReadLineAsync())!=null){var saved=line;App?.Invoke(()=>Receive(saved));}App?.Invoke(()=>{busy=false;waiting=false;status.Text="Ядро завершено · введите команду для перезапуска";});});
            _=Task.Run(async ()=> {string? line;while((line=await process.StandardError.ReadLineAsync())!=null){var saved=line;App?.Invoke(()=>ShowLog(saved));}});
            return true;
        } catch(Exception ex){ShowLog("Не удалось запустить Python: "+ex.Message);return false;}
    }
    void Send<T>(T message){try{backend!.StandardInput.WriteLine(JsonSerializer.Serialize(message));backend.StandardInput.Flush();}catch(Exception ex){busy=false;waiting=false;status.Text="Связь с ядром потеряна: "+ex.Message;}}
    void Receive(string line) {
        try {
            using var doc=JsonDocument.Parse(line);var msg=doc.RootElement;
            var kind=msg.GetProperty("type").GetString();var text=msg.TryGetProperty("text",out var t)?t.GetString()??"":"";
            switch(kind) {
                case "log":ShowLog(text);break;
                case "prompt":waiting=true;ShowLog(text);status.Text="Ожидается ваш ответ · Enter подтвердить ввод";input.SetFocus();break;
                case "progress":
                    progress.Visible=true;status.Text=text;
                    if(msg.TryGetProperty("ratio",out var r)&&r.ValueKind==JsonValueKind.Number)progress.Fraction=(float)r.GetDouble();
                    else {progress.Fraction=0;}
                    break;
                case "done":busy=false;waiting=false;progress.Visible=false;status.Text=text;input.SetFocus();break;
                case "context":context.Text="SHOTSMITH  /  "+text;break;
            }
        } catch(JsonException){ShowLog(line);}
    }
    void Exit(){if(busy){status.Text="Команда ещё выполняется. Дождитесь завершения; окно остаётся открытым.";return;}try{backend?.StandardInput.Close();}catch{}App?.RequestStop();}
    protected override void Dispose(bool disposing){if(disposing){if(backend is {HasExited:false}&&!busy){try{backend.StandardInput.Close();backend.WaitForExit(2000);}catch{}}backend?.Dispose();}base.Dispose(disposing);}
}
