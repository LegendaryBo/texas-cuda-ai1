package pl.wroc.uni.ii.evolution.servlets.managment;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintWriter;

import javax.servlet.ServletConfig;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import pl.wroc.uni.ii.evolution.servlets.databasesupport.structure.EvTaskInfo;
import pl.wroc.uni.ii.evolution.servlets.managment.communication.EvManagmentServletCommunicationImpl;

import com.oreilly.servlet.MultipartRequest;
import com.oreilly.servlet.multipart.DefaultFileRenamePolicy;

public class EvDistributionManagmentWWW extends HttpServlet {

  /**
   * 
   */
  private static final long serialVersionUID = -8288963843659147657L;

  private EvManagmentServletCommunicationImpl managment_servlet;

  private String managment_url;

  private String dir;


  @Override
  protected void doGet(HttpServletRequest request, HttpServletResponse response)
      throws ServletException, IOException {

    Object logged = request.getSession().getAttribute("logged_in");

    if (logged == null || ((Boolean) logged) == false) {
      response.sendRedirect("index.jsp");
    }

    String cmd = request.getParameter("cmd");
    String id_string = request.getParameter("id");
    if (cmd != null && id_string != null) {
      Integer id = Integer.parseInt(id_string);
      try {
        managment_servlet =
            new EvManagmentServletCommunicationImpl(managment_url);
        if (cmd.equals("stop")) {

          managment_servlet.stopTask(id);
        }
        if (cmd.equals("resume")) {

          managment_servlet.resumeTask(id);
        }
        if (cmd.equals("del")) {

          managment_servlet.deleteTask(id);
        }

        if (cmd.equals("clear")) {
          managment_servlet.clearTask(id);
        }

        if (cmd.equals("download")) {

          EvTaskInfo info = managment_servlet.getEvTask(id, true);

          response.setContentType("application/octet-stream");
          response.setHeader("content-disposition", "attachment; filename="
              + info.getDescription());

          OutputStream out = response.getOutputStream();

          out.write((byte[]) info.getJar());

          out.flush();
          return;

        }

      } catch (Exception ex) {

      }

    }

    response.setContentType("text/html");
    PrintWriter out = new PrintWriter(response.getWriter());

    producePage(out);

    out.flush();

  }


  private void producePage(PrintWriter out) {

    String tmp =
        "<html><head><meta http-equiv=\"Content-Type\" content=\"text/html; charset=UTF-8\"><title>wEvo</title><meta http-equiv=\"Content-Type\" content=\"text/html; charset=UTF-8\" /><link rel=\"stylesheet\" href=\"style.css\" type=\"text/css\" /></head><body> <a href=\"logout.jsp\"> Logout </a>  <h1> wEvo - Distributed Evolutionary Framework </h1>";

    tmp += "<h2>Task List </h2>";
    tmp +=
        "<table>  <th> ID </th> <th> DESCRIPTION </th> <th> STATUS </th> <th> NODES </th> <th> SUBMISSION TIME </th> <th> OPERATION </th> </tr> ";

    managment_servlet = new EvManagmentServletCommunicationImpl(managment_url);

    try {
      int[] ids = managment_servlet.getEvTaskIds();

      if (ids == null) {
        tmp +=
            "<td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>";
      }

      for (int id : ids) {
        EvTaskInfo info = managment_servlet.getEvTask(id, false);
        tmp += "<tr><td>" + id + "</td><td>" + info.getDescription() + "</td>";

        if (info.getStatus() == 1) {
          tmp += "<td>Submitted</td>";
        } else {
          tmp += "<td>Stopped</td>";
        }

        tmp +=
            "<td>" + managment_servlet.getNodeCountForTask(id) + "</td><td>"
                + info.getSubmissionTime() + "</td>";

        tmp +=
            "<td> <a href='?cmd=del&id=" + info.getId() + "'> delete </a>"
                + "&nbsp;&nbsp;&nbsp;<a href='?cmd=clear&id=" + info.getId()
                + "'> clear </a>" + "&nbsp;&nbsp;&nbsp;<a href='?cmd=stop&id="
                + info.getId() + "'> stop </a>"
                + "&nbsp;&nbsp;<a href='?cmd=resume&id=" + info.getId()
                + "'> resume </a>" + "&nbsp;&nbsp;<a href='?cmd=download&id="
                + info.getId() + "'> download </a>";

        tmp += "</tr>";
      }

    } catch (Exception e) {

    }

    tmp += "</table><br> <br>";

    tmp += "<h2> Actions </h2>";
    tmp += "<br> <a href=\"newtask.jsp\"> Add new task </a>";
    tmp +=
        "<br> <br><a href=\"worker.jsp\">Start worker as evolutionary algorithm</a>";
    tmp +=
        "<br> <br><a href=\"eval_worker.jsp\">Start worker as objective function evaluator</a>";
    tmp += "<br> <br><a href=\"charts.jsp\">View charts</a>";

    out.println(tmp);

  }


  @Override
  protected void doPost(HttpServletRequest request, HttpServletResponse response)
      throws ServletException, IOException {

    File directory = new File(dir);
    directory.mkdir();

    MultipartRequest mreq =
        new MultipartRequest(request, dir, 16 * 1024 * 1024,
            new DefaultFileRenamePolicy());
    String path = dir + "/" + mreq.getFilesystemName("file");
    String desc = mreq.getParameter("desc");

    InputStream in = new FileInputStream(path);
    byte[] file = new byte[in.available()];
    in.read(file);
    in.close();

    File f = new File(path);
    f.delete();

    managment_servlet = new EvManagmentServletCommunicationImpl(managment_url);

    try {
      managment_servlet.addTask(file, desc);
    } catch (Exception e) {
      response.getWriter().println(e);
    }

    doGet(request, response);
  }


  @Override
  public void init(ServletConfig conf) throws ServletException {

    managment_url = conf.getInitParameter("managment_servlet_url");
    dir = conf.getInitParameter("dir");
  }

}
