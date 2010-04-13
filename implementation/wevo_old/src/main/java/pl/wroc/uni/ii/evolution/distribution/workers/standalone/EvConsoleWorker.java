package pl.wroc.uni.ii.evolution.distribution.workers.standalone;

import pl.wroc.uni.ii.evolution.distribution.workers.EvEvolutionInterface;
import pl.wroc.uni.ii.evolution.distribution.workers.EvTaskLoaderImpl;
import pl.wroc.uni.ii.evolution.distribution.workers.EvTaskMaster;
import pl.wroc.uni.ii.evolution.distribution.workers.EvJARCacheImpl;
import pl.wroc.uni.ii.evolution.servlets.managment.communication.EvManagmentServletCommunication;
import pl.wroc.uni.ii.evolution.servlets.managment.communication.EvManagmentServletCommunicationImpl;

public class EvConsoleWorker {

  /**
   * @param args
   */
  public static void main(String[] args) {

    String tomcat_url = "";

    try {
      tomcat_url = args[0];
    } catch (Exception ex) {
      System.out.println("Usage: java -jar wevo_client.jar <tomcat_url>");
      return;
    }

    String managment_servlet_url =
        tomcat_url + "/wevo_system/DistributionManager";

    /** create object used for communication with servlets */
    EvManagmentServletCommunication proxy =
        new EvManagmentServletCommunicationImpl(managment_servlet_url);
    EvTaskLoaderImpl loader = new EvTaskLoaderImpl(tomcat_url);

    EvJARCacheImpl jar_manager = new EvJARCacheImpl(proxy);
    jar_manager.init(System.currentTimeMillis());

    EvTaskMaster interaction =
        new EvTaskMaster(proxy, loader, jar_manager, 2000, 0,
            (EvEvolutionInterface) null);

    /** start interaction with managment servlet */
    interaction.run();

  }
}