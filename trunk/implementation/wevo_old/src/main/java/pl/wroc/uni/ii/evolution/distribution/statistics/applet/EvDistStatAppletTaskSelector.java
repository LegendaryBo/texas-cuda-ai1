package pl.wroc.uni.ii.evolution.distribution.statistics.applet;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.io.IOException;

import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JPanel;

import pl.wroc.uni.ii.evolution.servlets.databasesupport.communication.EvDBServletCommunication;

/**
 * Component which shows and allow to select task id of stored statistics
 * 
 * @author Kacper Gorski
 */
public class EvDistStatAppletTaskSelector extends JPanel implements
    ActionListener, ItemListener {

  private static final long serialVersionUID = 1L;

  private JLabel title = new JLabel("Select tasks ID:");

  private JComboBox tasks = new JComboBox();

  private JButton refresh = new JButton("Refresh");

  private EvDBServletCommunication data_base;

  private EvDistStatAppletDomainSelector domain_selector = null;


  public EvDistStatAppletTaskSelector(EvDBServletCommunication data_base,
      EvDistStatAppletDomainSelector domain_selector) {
    this.data_base = data_base;
    this.domain_selector = domain_selector;
    setLayout(null);
    // positioning subcomponents
    title.setBounds(0, 0, 100, 20);
    add(title);
    tasks.setBounds(50, 20, 50, 20);
    add(tasks);
    refresh.setBounds(100, 20, 100, 20);
    add(refresh);
    refresh.addActionListener(this);
    tasks.addItemListener(this);

  }


  // refresh tasks shown in the component using connection given in contructor
  public void refreshId() {
    try {
      // removing old id
      tasks.removeAllItems();

      Long[] ids = data_base.getTaskIdsWithStatistics();

      // adding new ids
      if (ids == null) {
        tasks.addItem("No tasks with stats");
      } else {
        for (int i = 0; i < ids.length; i++) {
          tasks.addItem("" + ids[i]);
        }
      }
      tasks.setSelectedIndex(0);
    } catch (IOException e) {
      e.printStackTrace();
    }

  }


  // casted when clicked 'refresh'
  public void actionPerformed(ActionEvent e) {
    refreshId();
  }


  /**
   * Return selected task id by the component. Return -1 if no task selected.
   * 
   * @return selected task
   */
  public int getTaskId() {
    if (tasks.getSelectedItem() == null
        || ((String) tasks.getSelectedItem()).equals("No tasks with stats"))
      return -1; // when nothing selected
    else
      return Integer.parseInt((String) tasks.getSelectedItem()); // return
                                                                  // selected if
  }


  public void itemStateChanged(ItemEvent e) {
    domain_selector.refresh(getTaskId());
  }

}
