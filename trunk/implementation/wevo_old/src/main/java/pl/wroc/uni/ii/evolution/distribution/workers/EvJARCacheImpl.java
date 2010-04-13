package pl.wroc.uni.ii.evolution.distribution.workers;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.Enumeration;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import pl.wroc.uni.ii.evolution.servlets.managment.communication.EvManagmentServletCommunication;

/**
 * Local cache of JAR files with task.
 * 
 * @author Marcin Golebiowski
 */

public class EvJARCacheImpl implements EvJARCache {

  private EvManagmentServletCommunication manager = null;

  private String path;


  /**
   * @param manager_url url to Wevo Manager Servlet
   */
  public EvJARCacheImpl(EvManagmentServletCommunication manager) {
    this.manager = manager;
  }


  /**
   * Inits EvJARManager. Creates a storage directory.
   */
  public void init(long time) {
    /** Create path to storage directory * */
    path = System.getProperty("user.home");
    path = path.replace("\\", "/");
    path += "/wevo_tasks";

    File f = new File(path);
    deleteDir(f);

    File c = new File(path);
    c.mkdir();

    f = new File(path + "/" + time);
    f.mkdir();

    path += "/" + time;
  }


  /** Returns URL to local JAR file for given node and task * */
  public synchronized String getJARUrl(long node_id, long task_id) {

    /** Check if there is a jar and libs for this task* */
    File dir = new File(path + "/" + task_id);
    if (!dir.isDirectory()) {
      fetchTask(node_id, task_id);
    }
    return "file:/" + path + "/" + task_id + "/task.jar";
  }


  /** Remove all JARs * */
  public void dispose() {
    File directory = new File(path);
    deleteDir(directory);
  }


  private void fetchTask(long node_id, long task_id) {
    try {
      byte[] tmp = manager.getJAR(node_id);
      File directory = new File(path + "/" + task_id);
      directory.mkdir();

      FileOutputStream out =
          new FileOutputStream(path + "/" + task_id + "/task.jar");
      out.write(tmp);
      out.flush();
      out.close();

      unpackLibs(path + "/" + task_id + "/task.jar", path + "/" + task_id);

    } catch (Exception e) {
      e.printStackTrace();
    }
  }


  /**
   * Unpacks libs to given directory
   * 
   * @param jar_url
   * @param dir
   */
  private void unpackLibs(String jar_url, String dir) {
    try {
      JarFile jar = new JarFile(jar_url);
      Enumeration<JarEntry> entries = jar.entries();

      while (entries.hasMoreElements()) {
        JarEntry curr = entries.nextElement();
        String full_name = curr.getName();

        // System.out.println("Unpacking: " + full_name);

        String[] parts = full_name.split("/");

        if (full_name.matches("libs/.*") || full_name.matches("lib/.*")) {

          // System.out.println("Saving ...");
          InputStream imp = jar.getInputStream(curr);

          FileOutputStream f_out =
              new FileOutputStream(dir + "/" + parts[parts.length - 1]);

          int val = imp.read();
          while (val != -1) {
            f_out.write(val);
            val = imp.read();
          }
          f_out.flush();
          f_out.close();
          imp.close();
        }
      }
      jar.close();

    } catch (Exception e) {
      e.printStackTrace();
    }
  }


  private void deleteDir(File dir) {
    String path = dir.getAbsolutePath();

    if (dir.isDirectory()) {
      String[] children = dir.list();
      for (int i = 0; i < children.length; i++) {
        deleteDir(new File(path + "/" + children[i]));
      }
    }

    dir.delete();
    // The directory is now empty so delete it
  }


  public void clean() {
    deleteDir(new File(path));
  }

}
