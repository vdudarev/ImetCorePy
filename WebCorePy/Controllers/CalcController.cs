using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Configuration;
using Newtonsoft.Json.Linq;
using WebCorePy.Models;
using WebCorePy.Utils;

namespace WebCorePy.Controllers
{
    [Authorize(Roles = "User,PowerUser,Administrator")]
    public class CalcController : Controller
    {
        private static ConcurrentDictionary<int, Process> userProcess = new ConcurrentDictionary<int, Process>();
        private static ConcurrentDictionary<int, Timer> userTimer = new ConcurrentDictionary<int, Timer>();

        private Encoding encodingBatDos = Encoding.GetEncoding(866);    // for BAT and /py/!algorithms.template.json

        //private static Timer timer;
        const int timerInterval = 5000;

        private int _userId = 0;

        private int userId { 
            get {
                if (_userId == 0){
                    _userId = HttpContext.GetUserId();
                }
                return _userId;
            }
        }

        private static string[] ext = { "xls", "xlsx", "csv", "txt" };
        private static object lockObj = new object();
        IWebHostEnvironment env { get; }
        IConfiguration config { get; }
        public CalcController(IWebHostEnvironment env, IConfiguration config) {
            this.env = env;
            this.config = config;
        }


        public IActionResult Index()
        {
            GetAlgorithmsHtml();
            return View(userId);
        }

        public IActionResult Format()
        {
            return View();
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }

        public IActionResult Clear()
        {
            HttpContext.Session.Clear();
            RemoveAllAlgorythmsCookie();

            //AppDomain dom = Thread.GetDomain();

            //ProcessThreadCollection currentThreads = Process.GetCurrentProcess().Threads;
            //foreach (ProcessThread thread in currentThreads)    
            //{
            //    if (thread.Id == Thread.CurrentThread.ManagedThreadId)
            //        continue;
            //    thread.
            //   // Do whatever you need
            //}

            if (userProcess.TryGetValue(userId, out Process process)) {
                process?.Kill();
                userProcess.TryRemove(userId, out process);
            }
            if (userTimer.TryRemove(userId, out Timer timer))
            {
                Log($"Clear: userTimer.TryRemove executed successfully for userId: {userId}");
            }
            else
            {
                Log($"Clear: userTimer.TryRemove returned FALSE for userId: {userId}");
            }

            CleanUp(env.WebRootPath, userId);
            DirectoryInfo di = new DirectoryInfo(GetUserUploadFolder(env.WebRootPath, userId));
            foreach (FileInfo file in di.GetFiles())
            {
                if (file.Extension != ".exe") {
                    file.Delete();
                }
            }
            foreach (DirectoryInfo dir in di.GetDirectories())
            {
                dir.Delete(true);
            }
            return Redirect("~/calc/");
        }

        /*
        /// <summary>
        /// https://docs.microsoft.com/ru-ru/aspnet/core/mvc/models/file-uploads?view=aspnetcore-3.0
        /// </summary>
        /// <param name="files"></param>
        /// <returns></returns>
        [HttpPost("UploadFiles")]
        public async Task<IActionResult> Post(List<IFormFile> files)        // поменять сходу не получилось (IFormFile fileSingle)
        {
            long size = files.Sum(f => f.Length);

            // full path to file in temp location
            var filePath = Path.GetTempFileName();

            foreach (var formFile in files)
            {
                if (formFile.Length > 0)
                {
                    using (var stream = new FileStream(filePath, FileMode.Create))
                    {
                        await formFile.CopyToAsync(stream);
                    }
                }
            }

            // process uploaded files
            // Don't rely on or trust the FileName property without validation.

            return Ok(new { count = files.Count, size, filePath }); // {"count":1,"size":18506,"filePath":"C:\\Users\\Дударев Виктор\\AppData\\Local\\Temp\\tmp3CAF.tmp"}
        }
        */


        public string Msg { get; set; }


        /// <summary>
        /// сохранение файла в папке
        /// https://docs.microsoft.com/ru-ru/aspnet/core/mvc/models/file-uploads?view=aspnetcore-3.0
        /// </summary>
        /// <param name="folder">папка</param>
        /// <param name="type">0 - fileTrain, 1 - filePredict</param>
        /// <param name="file">файл </param>
        /// <returns></returns>
        private async Task<string> SaveFile(string folder, int type, List<IFormFile> file) {
            string filePath, fileName = null;
            long size = 0;
            foreach (var formFile in file)
            {
                fileName = formFile.FileName;
                filePath = System.IO.Path.Combine(folder, fileName);  // Path.GetTempFileName();
                if (formFile.Length > 0)
                {
                    using (var stream = new FileStream(filePath, FileMode.Create))
                    {

                        await formFile.CopyToAsync(stream);
                        size = formFile.Length;
                    }
                }
            }
            if (size > 0) {
                if (type == 0)
                    HttpContext.Session.SetString("fileTrain", fileName);
                if (type == 1)
                    HttpContext.Session.SetString("filePredict", fileName);
                return $"{fileName} ({size} байт)";
            }
            return null;
        }


        private string GetAlgorithmName(string json) {
            int i1 = json.IndexOf("\"name\": \"");
            if (i1 < 0)
                return null;
            int i2 = json.IndexOf("\"", i1 + 9);
            if (i2 < 0)
                return null;
            return json.Substring(i1 + 9, i2 - i1 - 9);
        }


        public string[] algorithms = null;
        public string[] Algorithms {
            get {
                if (algorithms == null)
                    algorithms = System.IO.File.ReadAllText(env.WebRootPath + "\\py\\!algorithms.template.json", encodingBatDos).Split(Environment.NewLine, StringSplitOptions.RemoveEmptyEntries);
                return algorithms;
            }
        }



        private string PrepareCookieName(string cookieName) {
            return cookieName.Replace(" ", "_").Replace("(", "_").Replace(")", "_").Replace(":", "_");
        }

        private string GetCookieName(string name) => $"alg_{PrepareCookieName(name)}";

        private string GetCookieForAlgorithm(string name) => Request?.Cookies?[GetCookieName(name)]?.ToString();


        /// <summary>
        /// обновление настроек алгоритмов 
        /// </summary>
        private void UpdateAlgorithmCookiesByRequestForm() {
            string[] arr = Algorithms;
            string key, name;
            for (int i = 0; i < arr.Length; i++) {
                name = GetAlgorithmName(arr[i]);
                key = GetCookieName(name);
                if (string.CompareOrdinal(Request?.Form[key], "1") == 0)
                    SetCookie(key, "1", 365);
                else
                    RemoveCookie(key);
            }
        }


        public string GetAlgorithmsHtml() {
            StringBuilder sb = new StringBuilder();
            string[] arr = Algorithms;
            string name, checkedStr;
            int MethodsSelectedCount = 0;
            for (int i = 0; i < arr.Length; i++) {
                name = GetAlgorithmName(arr[i]);
                if (string.IsNullOrEmpty(GetCookieForAlgorithm(name)))
                    checkedStr = string.Empty;
                else {
                    checkedStr = " checked=\"checked\"";
                    MethodsSelectedCount++;
                }
                sb.AppendLine($@"<div class=""input-group mb-3"">
  <div class=""input-group-prepend"">
    <div class=""input-group-text"">
      <input type=""checkbox"" value=""1"" name=""{GetCookieName(name)}"" aria-label=""{name}"" class=""me-2"" {checkedStr}> {name}
    </div>
  </div>
  <input type=""text"" class=""form-control"" aria-label=""{name}"" value='{arr[i]}'>
</div>");
            }
            ViewBag.MethodsSelectedCount = MethodsSelectedCount.ToString();
            return ViewBag.Methods = sb.ToString();
        }



        private string GetAllAlgorithmsJson() {
            return string.Join($",{Environment.NewLine}", Algorithms);
        }

        private string GetSelectedAlgorithmsJson() {
            List<string> list = new List<string>();
            foreach (var item in Algorithms) {
                string name = GetAlgorithmName(item);
                //if (!string.IsNullOrEmpty(GetCookieForAlgorithm(name)))
                if (string.CompareOrdinal(Request?.Form[GetCookieName(name)], "1") == 0)
                    list.Add(item);
            }
            return string.Join($",{Environment.NewLine}", list);
        }


        private string GetSettingsJson(int timeout4Method, string algorithmsJson=null) {
            string jsonTemplate = System.IO.File.ReadAllText(env.WebRootPath + "\\py\\!settings.template.json", encodingBatDos);
            if (algorithmsJson==null)
                algorithmsJson = GetSelectedAlgorithmsJson();
            return jsonTemplate
                .Replace("#ALGORITHMS#", algorithmsJson)
                .Replace("#UserUpload#", GetUserUploadFolder(env.WebRootPath, userId).Replace("\\", "\\\\"))
                .Replace("#ROOT#", env.WebRootPath.Replace("\\", "\\\\"))
                .Replace("#ROOTRightSlash#", env.WebRootPath.Replace("\\", "/"))
                .Replace("#fileTrain#", HttpContext.Session.GetString("fileTrain"))
                .Replace("#filePredict#", HttpContext.Session.GetString("filePredict"))
                .Replace("#TIMEOUT#", timeout4Method.ToString())
                ;
        }

        private string GetRunBatTemplate()
        {
            string strTemplate = System.IO.File.ReadAllText(env.WebRootPath + "\\py\\!run.template.bat", encodingBatDos);
            string pythonExe = config.GetValue<string>("AppSettings:PythonExePath");
            strTemplate = strTemplate.Replace("#UserUpload#", GetUserUploadFolder(env.WebRootPath, userId).Replace("\\", "\\\\"));
            strTemplate = strTemplate.Replace("#WebRootPath#", env.WebRootPath);
            strTemplate = strTemplate.Replace("#PythonExe#", pythonExe);
            return strTemplate;
        }

        /// <summary>
        /// Основные вычисления
        /// </summary>
        /// <param name="fileTrain">файл для обучения</param>
        /// <param name="filePredict">файл для тестирования</param>
        /// <returns></returns>
        [HttpPost("UploadFiles")]
        public async Task<IActionResult> Post(List<IFormFile> fileTrain, List<IFormFile> filePredict, string timeout)        // поменять сходу не получилось (IFormFile fileSingle)
        {
            if (userProcess.TryGetValue(userId, out Process process) && process != null)
            {
                ViewBag.Msg = $"<div class=\"alert alert-danger\" role=\"alert\">Идет обработка... Для ее принудительного завершения перейдите по ссылке \"Очистить сессию\"</div>";
                return View("Index", userId);
            }


            HttpContext.Session.Remove("fileTrain");
            HttpContext.Session.Remove("filePredict");
            UpdateAlgorithmCookiesByRequestForm();  // сохраняем выбор алгоритмов в куки
            //GetAlgorithmsHtml();

            ViewBag.ShowResults = false;
            ViewBag.ShowResultsXls = false;
            int timeout4Method;
            int.TryParse(timeout, out timeout4Method);
            long size = fileTrain.Sum(f => f.Length);

            string folder = GetUserUploadFolder(env.WebRootPath, userId);
            if (!Directory.Exists(folder)) { 
                Directory.CreateDirectory(folder);
            }


            string filePath = string.Empty;
            string fileName = string.Empty;
            string[] msg = new string[2];
            msg[0] = await SaveFile(folder, 0, fileTrain);
            msg[1] = await SaveFile(folder, 1, filePredict);
            if (string.IsNullOrEmpty(msg[0]) || string.IsNullOrEmpty(msg[1])) {
                ViewBag.Msg = $"<div class=\"alert alert-success\" role=\"alert\">Загружен файл {string.Join(" и ", msg.Where(s => !string.IsNullOrEmpty(s)))}</div>";
            }

            if (string.IsNullOrEmpty(HttpContext.Session.GetString("fileTrain"))) {
                ViewBag.Msg = $"<div class=\"alert alert-danger\" role=\"alert\">Не выбран файл для обучения!</div>";
                return View("Index", userId);
            }

            // process uploaded files
            //return Ok(new { count = files.Count, size, filePath, WebRootPath = env.WebRootPath }); // {"count":1,"size":18506,"filePath":"C:\\Users\\Дударев Виктор\\AppData\\Local\\Temp\\tmp3CAF.tmp"}

            string runBat = folder + "\\run.bat";
            //prepareBat();
            string py = config.GetValue<string>("AppSettings:AnacondaPath");
            //System.Text.Encoding.RegisterProvider(System.Text.CodePagesEncodingProvider.Instance);    // ???????
            // %root%\python.exe "D:\WWW\IMET\IMETCorePy\WebCorePy\wwwroot\py\regressor.py" "D:\WWW\IMET\IMETCorePy\WebCorePy\wwwroot\upload\!settings.json"


            System.IO.File.WriteAllText(folder + "\\!settings.json",
                GetSettingsJson(timeout4Method),
                encoding: new UTF8Encoding(false)   // without BOM!
                );

            string batFileInstructions = GetRunBatTemplate();
            System.IO.File.WriteAllText(runBat, batFileInstructions, encoding: encodingBatDos);
            // запустим программку на питоне
            // RunPythonScript("\"" + env.WebRootPath + "\\py\\regressor.py\" \"" + HttpContext.Session.GetString("fileTrain") + "\" \"" + HttpContext.Session.GetString("filePredict") + "\" \"result.xls\" \"log.txt\"");
            RunCmd(runBat, string.Empty);
            ViewBag.ShowResultsXlsExtension = Path.GetExtension(HttpContext.Session.GetString("fileTrain")).Substring(1).ToLower();
            if (!string.IsNullOrEmpty(HttpContext.Session.GetString("filePredict"))) {
                ViewBag.ShowResultsXls = true;
            }
            return View("Index", userId);
        }


        private void Log(string message)
        {
            lock (lockObj)
            {
                System.IO.File.AppendAllText($@"{GetUserUploadFolder(env.WebRootPath, userId)}\!.txt", message + Environment.NewLine, encoding: Encoding.GetEncoding(1251));
            }
        }

        private void CleanUp(string WebRootPath, int userId)
        {
            if (Directory.Exists(GetUserUploadFolder(WebRootPath, userId) + @"\TestData"))
                Directory.Delete(GetUserUploadFolder(WebRootPath, userId) + @"\TestData", true);
            System.IO.File.Delete(GetUserUploadFolder(WebRootPath, userId) + @"\!.txt");
            System.IO.File.Delete(GetUserUploadFolder(WebRootPath, userId) + @"\log.txt");
            Array.ForEach(ext, x => {
                System.IO.File.Delete(GetUserUploadFolder(WebRootPath, userId) + @"\result." + x);
                System.IO.File.Delete(GetUserUploadFolder(WebRootPath, userId) + @"\resulеScore." + x);
            });
        }

        private void TimerCallback(object o)
        {
            (string WebRootPath, int userId) = ((string, int))o;
            DateTime dt = DateTime.Now.AddSeconds(-3);
            for (int i = 0; i < ext.Length; i++)
            {
                string path = GetUserUploadFolder(WebRootPath, userId) + @"\resultScore." + ext[i];
                DateTime fdt;
                if (System.IO.File.Exists(path) && (fdt = System.IO.File.GetLastWriteTime(path)) < dt)
                {
                    if (userTimer.TryGetValue(userId, out Timer timer))
                    {
                        timer?.Dispose();
                        if (userTimer.TryRemove(userId, out timer))
                        {
                            Log($"TimerCallback: userTimer.TryRemove executed successfully for userId: {userId}");
                        }
                        else {
                            Log($"TimerCallback: userTimer.TryRemove returned FALSE for userId: {userId}");
                        }
                    }
                    Log($"{DateTime.Now} Killing process since file resultScore.{ext[i]} was written at {fdt}");
                    if (userProcess.TryGetValue(userId, out Process process))
                    {
                        process?.Kill();
                        if (userProcess.TryRemove(userId, out process))
                        {
                            Log($"TimerCallback: userProcess.TryRemove executed successfully for userId: {userId}");
                        }
                        else {
                            Log($"TimerCallback: userProcess.TryRemove returned FALSE for userId: {userId}");
                        }
                    }
                    Log($"{DateTime.Now} Process killed...");
                    return;
                }
            }
            Log($"{DateTime.Now} TimerCallback pulse...");
        }

        private void RunCmd(string cmd, string args)
        {
            CleanUp(env.WebRootPath, userId);
            Log($"RunCmd START: {DateTime.Now}");
            ProcessStartInfo start = new ProcessStartInfo
            {   CreateNoWindow = true,
                FileName = cmd,
                Arguments = args,
                UseShellExecute = false,
                // DO NOT USE RedirectStandardXXX if they are not read (4096 bufer size => overflow == deadlock) 
                // https://stackoverflow.com/questions/43204624/what-are-buffer-sizes-of-process-start-for-stdout-and-stderr
                RedirectStandardOutput = false,
                RedirectStandardError = false,  
                // StandardOutputEncoding = Encoding.GetEncoding(1251),
                // StandardErrorEncoding = Encoding.GetEncoding(1251)
            };
            string stdOut = string.Empty;
            string stdErr = string.Empty;
            // Task<string> taskStdOut = null;
            // Task<string> taskstdErr = null;
            Log($"Process beforeSTARTED: {DateTime.Now}");
            var tokenSource = new CancellationTokenSource();
            var token = tokenSource.Token;
            Process process;
            using (process = Process.Start(start))
            {
                if (!userProcess.TryAdd(userId, process)) {
                    Log($"RunCmd: userProcess.TryAdd returned FALSE for userId: {userId}");
                }
                Timer timer;
                using (timer = new Timer(TimerCallback, (env.WebRootPath, userId), timerInterval, timerInterval)) {
                    if (!userTimer.TryAdd(userId, timer))
                    {
                        Log($"RunCmd: userTimer.TryAdd returned FALSE for userId: {userId}");
                    }

                    // ВНИМАНИЕ Это нельзя раскомментировать, т.к. если процесс убивается (kill), то здесь на reader.ReadToEnd(); мы повисаем...
                    //using (StreamReader reader = process.StandardOutput)
                    //{
                    //    stdOut = reader.ReadToEnd();
                    //}
                    //using (StreamReader reader = process.StandardError)
                    //{
                    //    stdErr = reader.ReadToEnd();
                    //}

                    //using (StreamReader reader = process.StandardOutput)
                    //{
                    //    taskStdOut = reader.ReadToEndAsync();
                    //}
                    //using (StreamReader reader = process.StandardError)
                    //{
                    //    taskstdErr = reader.ReadToEndAsync();
                    //}

                    Log($"Process STARTED: {DateTime.Now}");
                    process.WaitForExit();
                    Log($"Process ENDED: {DateTime.Now}");
                }
                if (!userTimer.TryRemove(userId, out timer))
                {
                    Log($"RunCmd: userTimer.TryRemove returned FALSE for userId: {userId}");
                }
                timer = null;
            }
            //if (taskStdOut.IsCompleted)
            //{
            //    stdOut = taskStdOut.Result;
            //}
            //else
            //    taskStdOut.Dispose();   // нелья Dispose активную задачу!
            //if (taskstdErr.IsCompleted)
            //{
            //    stdErr = taskstdErr.Result;
            //}
            //else
            //    taskstdErr.Dispose();
            if (!userProcess.TryRemove(userId, out process)) {
                Log($"RunCmd: userProcess.TryRemove returned FALSE for userId: {userId}");
            }
            process = null;
            ViewBag.ShowResults = true;
            // надо использовать кодировку 1251
            ViewBag.Log = System.IO.File.ReadAllText($@"{GetUserUploadFolder(env.WebRootPath, userId)}\log.txt", encoding: Encoding.GetEncoding(1251))
                .Replace("\r\n", "<br>")
                // + "<hr><b>отладочная информация</b><hr>"
                // + "<b>stdOut</b>: " + stdOut + "<br>"
                // + "<b>stdErr</b>: " + stdErr + "<br>"
                ;
            Log($"RunCmd END");
        }
        
        /// <summary>
        /// Get Upload Folder For particular User (considering Task)
        /// </summary>
        /// <param name="userId"></param>
        /// <param name="taskId"></param>
        /// <returns></returns>
        private string GetUserUploadFolder(string rootPath, int userId, int taskId = 0) {
            //return $@"{rootPath}\Upload\User{userId}";
            return $@"{rootPath}{UserUtils.GetUserUploadRelativeFolder(userId, taskId).Replace("/", "\\")}";
        }


        /// <summary>
        /// получение параметра из куков
        /// </summary>
        /// <param name="str">имя параметра</param>
        /// <returns>строка (или null)</returns>
        public string GetCookie(string key) {
            if (HttpContext.Request.Cookies.TryGetValue(key, out string value))
                return value;
            return null;
        }

        /// <summary>
        /// записываем кук 
        /// </summary>
        /// <param name="CookieName">имя кука</param>
        /// <param name="CookieValue">значение кука</param>
        /// <param name="Days">на сколько дней? (0 - сессионный кук)</param>
        public void SetCookie(string CookieName, string CookieValue, int Days=0) {
            if (Days > 0)
                HttpContext.Response.Cookies.Append(CookieName, CookieValue, new CookieOptions() { Expires = DateTime.Now.AddDays(Days) });
            else
                HttpContext.Response.Cookies.Append(CookieName, CookieValue);
        }


        /// <summary>
        /// удалить кук
        /// </summary>
        /// <param name="CookieName"></param>
        public void RemoveCookie(string CookieName) {
            if (HttpContext.Request.Cookies.ContainsKey(CookieName)) {
                HttpContext.Response.Cookies.Delete(CookieName);
            }
        }



        /// <summary>
        /// удалить куки алгоритмов
        /// </summary>
        public void RemoveAllAlgorythmsCookie()
        {
            string name;
            foreach (string key in Algorithms)
            {
                name = GetCookieName(GetAlgorithmName(key));
                RemoveCookie(name);
            }
        }
    }
}
