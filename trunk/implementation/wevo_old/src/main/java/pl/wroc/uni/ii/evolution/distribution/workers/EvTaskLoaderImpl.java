package pl.wroc.uni.ii.evolution.distribution.workers;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.net.JarURLConnection;
import java.net.URL;
import java.net.URLClassLoader;
import java.util.jar.Attributes;

import sun.misc.URLClassPath;

/**
 * It loads EvolutionaryTask from JAR. Manifest of this JAR must have attribute:
 * IslandCreator. Value of this attribute must be a name of class implementing
 * IslandCreator interface.
 * 
 * @author Piotr Lipinski, Marcin Golebiowski, Kamil Dworakowski, Kacper Gorski
 */
@SuppressWarnings("unchecked")
public class EvTaskLoaderImpl implements EvTaskLoader {

  /** class used for hide reflection interaction */
  class TaskReflectionWrapper implements Runnable {
    private Object task;


    public TaskReflectionWrapper(Object task) {
      this.task = task;
    }


    public void run() {
      Class task_class = task.getClass();

      Method method = null;
      try {
        method = task_class.getMethod("run", (Class[]) null);
        method.invoke(task, (Object[]) null);
      } catch (Exception e) {
        EvConsole.println(e.toString());
        e.printStackTrace(System.out);
      }
    }


    // to hide reflection
    public void setInterface(EvEvalTaskInterface inter) {
      Class task_class = task.getClass();

      Method method = null;
      try {

        method =
            task_class
                .getMethod(
                    "setInterface",
                    new Class[] {task
                        .getClass()
                        .getClassLoader()
                        .loadClass(
                            "pl.wroc.uni.ii.evolution.distribution.workers.EvEvalTaskInterface")});
        method.invoke(task, new Object[] {inter});

      } catch (Exception e) {
        // System.out.println(e.getMessage());
        EvConsole.println(e.toString());
        e.printStackTrace(System.out);
      }
    }


    public void setInterface(EvEvolutionInterface inter) {
      Class task_class = task.getClass();

      Method method = null;
      try {

        method =
            task_class
                .getMethod(
                    "setInterface",
                    new Class[] {task
                        .getClass()
                        .getClassLoader()
                        .loadClass(
                            "pl.wroc.uni.ii.evolution.distribution.workers.EvEvolutionInterface")});
        method.invoke(task, new Object[] {inter});

      } catch (Exception e) {
        // System.out.println(e.getMessage());
        EvConsole.println(e.toString());
        e.printStackTrace(System.out);
      }
    }

  }

  private String wevo_server_url;


  public EvTaskLoaderImpl(String wevo_server_url) {
    this.wevo_server_url = wevo_server_url;
  }


  /**
   * Returns instance of EvIsland that is created by some class in JAR file.
   * Name of this class is stored in JAR's manifest.
   * 
   * @param jar_url
   * @param task_id
   * @param node_id
   * @return EvolutionaryTask
   */
  public Runnable getTask(String jar_url, int task_id, long node_id, int type,
      EvEvalTaskInterface inter, EvEvolutionInterface evol_inter) {

    try {
      // EvConsole.println("Loading EvTask from:" + jar_url);
      /** read name of TaskCreator that is stored in JAR manifest */
      URL url = new URL("jar:" + jar_url + "!/");
      JarURLConnection uc = (JarURLConnection) url.openConnection();

      Attributes attr = uc.getMainAttributes();
      String creator_name = null;

      if (type == 0) {
        creator_name = attr.getValue("TaskCreator");
      }

      if (type == 1) {
        creator_name =
            "pl.wroc.uni.ii.evolution.distribution.tasks.EvEvaluatorTask";
      }

      // EvConsole.println("IslandCreator: <" + creator_name + ">");
      // EvConsole.println("jar_url:<" + jar_url + ">");

      /** load class from JAR */

      superDuperStrangeAddURLtoClasspatth(new URL(jar_url));
      Class creator = getClass().getClassLoader().loadClass(creator_name);

      // EvConsole.println("Jar loaded");
      /** create new instance of that class */
      Object obj = creator.newInstance();
      // EvConsole.println("Instance created");
      /** get 'create' method */
      Method m = null;

      m =
          creator.getMethod("create", new Class[] {int.class, long.class,
              String.class});

      m.setAccessible(true);
      // EvConsole.println("Create method extracted");

      /** create EvolutionaryTask */
      Object obj_task = null;
      obj_task =
          m.invoke(obj, new Object[] {task_id, node_id, wevo_server_url});
      // EvConsole.println("Task created");

      TaskReflectionWrapper task = new TaskReflectionWrapper(obj_task);
      if (type == 1)
        task.setInterface(inter);
      else
        task.setInterface(evol_inter);
      // EvConsole.println("Wrapped task created");
      return task;

    } catch (Exception ex) {
      EvConsole.println("LOAD TASK EXCEPTION: " + ex.toString());
      ex.printStackTrace(System.out);
      return null;
    }

  }


  /**
   * It's very stupid way to add external classes when running program into the
   * classpath. We use reflection to access URLClassPath of current
   * classLoader() and then P.S. If you know how to do it in normal way, plz
   * contact admin@34all.org - I have wasted so much time trying to do this
   * $#%#@$ !!!
   * 
   * @param url containing resources (jar or directory with classed or jars) to
   *        add to classpath
   */
  private void superDuperStrangeAddURLtoClasspatth(URL url) {
    try {
      Field field = (URLClassLoader.class).getDeclaredField("ucp");
      field.setAccessible(true);
      URLClassPath ucp =
          (URLClassPath) field
              .get((URLClassLoader) getClass().getClassLoader()); // oh yeah!
      ucp.addURL(url);
    } catch (Exception e) {
      e.printStackTrace();
    }
  }
}
