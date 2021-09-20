using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading;

namespace CSharpRunner4Python {
    class Program {

        static string WebRootPath;
        static Process process;
        static Timer timer;

        /// <summary>
        /// таймер проверки будет срабатывать каждые 5 секунд
        /// </summary>
        const int timerInterval = 5000;

        /// <summary>
        /// точка входа
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args) {
            Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);
            WebRootPath = $@"{Environment.CurrentDirectory}\DataFiles";
            string folder = $@"{WebRootPath}\Upload";
            string runBat = $@"{folder}\run.bat";
            // string runBat = folder + "\\runSleep.bat";
            // string runBat = folder + "\\runPythonSleep.bat";
            RunCmd(WebRootPath, runBat, string.Empty);
        }

        /// <summary>
        /// все возможные значения расширений
        /// </summary>
        static readonly string[] ext = { "xls", "xlsx", "csv", "txt" };

        /// <summary>
        /// очистка
        /// </summary>
        /// <param name="WebRootPath">путь к корню приложения на диске</param>
        private static void CleanUp(string WebRootPath) {
            File.Delete($@"{WebRootPath}\Upload\!.txt");
            File.Delete($@"{WebRootPath}\Upload\log.txt");
            Array.ForEach(ext, x => {
                File.Delete($@"{WebRootPath}\Upload\result.{x}");
                File.Delete($@"{WebRootPath}\Upload\resulеScore.{x}");
            });
        }


        /// <summary>
        /// наблюдаем за тем, как отрабатывает задание (бывает, что файл выходной правильный формируется, но питоновский процесс висит - не завершается)
        /// </summary>
        /// <param name="o"></param>
        private static void TimerCallback(object o)
        {
            string WebRootPath = o as string;
            DateTime dt = DateTime.Now.AddSeconds(-2);
            for (int i = 0; i < ext.Length; i++)
            {
                string path = $@"{WebRootPath}\Upload\result.{ext[i]}";
                DateTime fdt;
                if (File.Exists(path) && (fdt=File.GetLastWriteTime(path)) < dt) {
                    timer.Dispose();
                    Log($"{DateTime.Now} Killing process since file result.{ext[i]} was written at {fdt}{Environment.NewLine}");
                    process.Kill();
                    Log($"{DateTime.Now} Process killed...{Environment.NewLine}");
                    break;
                }
            }
        }

        /// <summary>
        /// запуск задания
        /// </summary>
        /// <param name="WebRootPath">путь к корню сайта (нужен, чтобы проверить готовность задания)</param>
        /// <param name="cmd">пусть к bat-файлу для запуска</param>
        /// <param name="args">параметры командной строки</param>
        private static void RunCmd(string WebRootPath, string cmd, string args) {
            CleanUp(WebRootPath);
            Log($"RunCmd START: {DateTime.Now}{Environment.NewLine}");
            ProcessStartInfo start = new ProcessStartInfo {
                CreateNoWindow = true,
                FileName = cmd,
                Arguments = args,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                StandardOutputEncoding = Encoding.GetEncoding(1251),
                StandardErrorEncoding = Encoding.GetEncoding(1251)
            };
            string stdOut = string.Empty;
            string stdErr = string.Empty;
            Log($"Process beforeSTARTED: {DateTime.Now}{Environment.NewLine}");
            using (process = Process.Start(start))
            {
                using (timer = new Timer(callback: TimerCallback, state: WebRootPath, dueTime: timerInterval, period: timerInterval)) {
                    Log($"Process STARTED: {DateTime.Now}{Environment.NewLine}");
                    process.WaitForExit();
                    //using (StreamReader reader = process.StandardOutput) {
                    //    stdOut = reader.ReadToEnd();
                    //}
                    //using (StreamReader reader = process.StandardError) {
                    //    stdErr = reader.ReadToEnd();
                    //}
                    Log($"Process ENDED: {DateTime.Now}\r\n");
                }
            }
            Log($"RunCmd END");
        }

        private static object lockObj = new object();
        private static void Log(string message) {
            lock (lockObj) {
                File.AppendAllText($@"{WebRootPath}\Upload\!.txt", message, encoding: Encoding.GetEncoding(1251));
            }
        }


    }
}
