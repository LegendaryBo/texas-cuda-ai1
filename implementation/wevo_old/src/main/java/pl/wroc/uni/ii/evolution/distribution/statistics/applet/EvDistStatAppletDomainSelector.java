package pl.wroc.uni.ii.evolution.distribution.statistics.applet;

import java.io.IOException;

import javax.swing.DefaultListModel;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.ListSelectionModel;
import javax.swing.event.ListSelectionEvent;
import javax.swing.event.ListSelectionListener;

import pl.wroc.uni.ii.evolution.servlets.databasesupport.communication.EvDBServletCommunication;

/**
 * Component on which you can select domain of the statistics (nodes, cells)
 * 
 * @author Kacper Gorski
 */
public class EvDistStatAppletDomainSelector extends JPanel implements
    ListSelectionListener {

  private static final long serialVersionUID = 1L;

  private JLabel title = new JLabel("Select statistics domain:");

  private JLabel title2 = new JLabel("Cell:");

  private JLabel title3 = new JLabel("Node:");

  private JComboBox nodes_list = new JComboBox();

  DefaultListModel data = new DefaultListModel();

  private JList cells_list = new JList(data);

  private EvDBServletCommunication db_comm;

  private Integer task_id = null;


  public EvDistStatAppletDomainSelector(EvDBServletCommunication db_comm) {

    this.db_comm = db_comm;
    setLayout(null);

    title.setBounds(0, 0, 150, 20);
    add(title);

    title2.setBounds(0, 22, 150, 20);
    add(title2);
    JScrollPane scroller = new JScrollPane(cells_list);
    scroller.setBounds(0, 40, 150, 50);
    add(scroller);
    cells_list.addListSelectionListener(this);

    title3.setBounds(200, 22, 150, 20);
    add(title3);
    nodes_list.setBounds(200, 40, 150, 20);
    add(nodes_list);

    cells_list.setSelectionMode(ListSelectionModel.SINGLE_INTERVAL_SELECTION);

  }


  /**
   * Updates subcomponents and fills them with nodes, cells ids
   * 
   * @param db - database
   * @param task_id
   */
  public void refresh(int task_id) {
    try {

      this.task_id = task_id;
      // removing old list of nodes und adding a new on

      Long[] cells = db_comm.getCellIdsWithStatistics(task_id);

      // removing old list of cells und adding a new one
      data.removeAllElements();

      if (cells != null) {
        for (int i = 0; i < cells.length; i++) {

          data.addElement(cells[i] + "");
          System.out.println("sdf");
        }
        cells_list.setSelectedIndex(0);
      }

    } catch (IOException e) {
      e.printStackTrace();
    }

  }


  public Long[] getCellId() {
    Object[] objects = cells_list.getSelectedValues();
    Long tab[] = new Long[objects.length];
    for (int i = 0; i < objects.length; i++) {
      tab[i] = Long.parseLong((String) objects[i]);
    }
    if (objects != null && objects.length > 0) {
      return tab;
    }
    return null;
  }


  public Long getNodeId() {
    Object[] objects = nodes_list.getSelectedObjects();
    if (objects != null && objects.length > 0) {
      return Long.parseLong((String) objects[0]);
    }
    return null;
  }


  public void valueChanged(ListSelectionEvent e) {
    Long[] cell = getCellId();

    if (task_id != null && cell != null) {

      Long[] nodes = null;
      try {
        nodes = db_comm.getNodesIdsWithStatistics(task_id, cell[0]);
      } catch (IOException e2) {
        e2.printStackTrace();
      }

      nodes_list.removeAllItems();
      if (nodes != null) {

        for (int j = 0; j < nodes.length; j++) {
          nodes_list.addItem(nodes[j] + "");
        }
        nodes_list.setSelectedIndex(0);
      }

    }

  }

}
