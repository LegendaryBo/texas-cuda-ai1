package pl.wroc.uni.ii.evolution.distribution.tools;

/**
 * @author Kacper Gorski (admin@34all.org) Some tools used in wevo enviroment.<BR>
 *         Icludes:<BR> - Simple tools to convert wevo server URL into direct
 *         servlets URL's<BR>
 */
public abstract class EvDistributionTools {

  /**
   * @param wevo_server_url
   * @return url of upload servlet
   */
  public static String upload_servlet_url(String wevo_server_url) {
    if (wevo_server_url.length() == 0)
      throw new IllegalArgumentException("wevo server url is empty!");

    if (wevo_server_url.endsWith("/"))
      return wevo_server_url + "wevo_databasesupport/UploadServlet";
    else
      return wevo_server_url + "/wevo_databasesupport/UploadServlet";
  }


  /**
   * @param wevo_server_url
   * @return url of download servlet
   */
  public static String download_servlet_url(String wevo_server_url) {
    if (wevo_server_url.length() == 0)
      throw new IllegalArgumentException("wevo server url is empty!");

    if (wevo_server_url.endsWith("/"))
      return wevo_server_url + "wevo_databasesupport/DownloadServlet";
    else
      return wevo_server_url + "/wevo_databasesupport/DownloadServlet";
  }


  /**
   * @param wevo_server_url
   * @return url of eval servlet
   */
  public static String eval_servlet_url(String wevo_server_url) {
    if (wevo_server_url.length() == 0)
      throw new IllegalArgumentException("wevo server url is empty!");

    if (wevo_server_url.endsWith("/"))
      return wevo_server_url + "wevo_eval/EvalMaster";
    else
      return wevo_server_url + "/wevo_eval/EvalMaster";
  }

}
