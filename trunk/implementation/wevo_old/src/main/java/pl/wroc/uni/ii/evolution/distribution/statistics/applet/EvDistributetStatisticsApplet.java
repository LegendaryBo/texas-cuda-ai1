package pl.wroc.uni.ii.evolution.distribution.statistics.applet;

import javax.swing.JApplet;
import javax.swing.JFrame;

import pl.wroc.uni.ii.evolution.servlets.databasesupport.communication.EvDBServletCommunication;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.communication.EvDBServletCommunicationImpl;

/**
 * Simple applet which allow you to display statistics of the selected task in
 * easy way.
 * 
 * @author Kacper Gorski
 */
public class EvDistributetStatisticsApplet extends JApplet {

  private static final long serialVersionUID = 1L;

  public EvDBServletCommunication db_comm = null;

  public EvDistStatAppletDomainSelector domain_selector = null;

  public EvDistStatAppletTaskSelector task_selector;

  private EvDistStatAppletTypeSelector type_selector = null;


  public static void main(String[] args) {
    JFrame frame = new JFrame();
    EvDistributetStatisticsApplet inst = new EvDistributetStatisticsApplet();
    frame.getContentPane().add(inst);
    frame.setVisible(true);
    inst.init();
  }


  public void init() {
    // String wevo_server_url = "http://127.0.0.1:8080";
    String wevo_server_url = getParameter("wevo_server_url");

    this.db_comm = new EvDBServletCommunicationImpl(wevo_server_url);

    this.domain_selector = new EvDistStatAppletDomainSelector(db_comm);
    this.task_selector =
        new EvDistStatAppletTaskSelector(db_comm, domain_selector);
    this.type_selector = new EvDistStatAppletTypeSelector(this);

    setLayout(null);

    task_selector.setBounds(0, 0, 600, 50);
    add(task_selector);
    domain_selector.setBounds(0, 50, 600, 100);
    add(domain_selector);
    type_selector.setBounds(0, 150, 600, 300);
    add(type_selector);

    domain_selector.setVisible(true);
    task_selector.setVisible(true);
    type_selector.setVisible(true);
    task_selector.refreshId();

  }


  public void connect(EvDBServletCommunication db) {
    task_selector.refreshId();
  }

}
