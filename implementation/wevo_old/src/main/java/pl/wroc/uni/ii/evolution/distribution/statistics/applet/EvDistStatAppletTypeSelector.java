package pl.wroc.uni.ii.evolution.distribution.statistics.applet;

import java.util.ArrayList;

import javax.swing.ButtonGroup;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import pl.wroc.uni.ii.evolution.distribution.statistics.applet.statTypes.EvAvgGenesValueStatPanel;
import pl.wroc.uni.ii.evolution.distribution.statistics.applet.statTypes.EvDistributedObjValueStat;
import pl.wroc.uni.ii.evolution.distribution.statistics.applet.statTypes.EvGenesChangeStatPanel;
import pl.wroc.uni.ii.evolution.distribution.statistics.applet.statTypes.EvGenesOriginStatPanel;
import pl.wroc.uni.ii.evolution.distribution.statistics.applet.statTypes.EvMaxAvgStatPanel;
import pl.wroc.uni.ii.evolution.distribution.statistics.applet.statTypes.EvStatisticPanel;
import pl.wroc.uni.ii.evolution.distribution.statistics.applet.statTypes.EvValueDistributionAreaStatPanel;

/**
 * Component which allows to select the type of statistics
 * 
 * @author Kacper Gorski
 */
public class EvDistStatAppletTypeSelector extends JPanel implements
    ChangeListener {

  private static final long serialVersionUID = 1L;

  // contains components, which are displayed when a proper radio button is
  // ticked
  private ArrayList<EvStatisticPanel> stats = new ArrayList<EvStatisticPanel>();

  private JLabel title = new JLabel("Select the type of statistic:");

  private ButtonGroup butt_group = new ButtonGroup();

  private JRadioButton[] radio_butt;


  public EvDistStatAppletTypeSelector(EvDistributetStatisticsApplet applet) {

    setLayout(null);
    title.setBounds(0, 0, 200, 20);
    add(title);

    stats.add(new EvAvgGenesValueStatPanel(applet));
    stats.add(new EvMaxAvgStatPanel(applet));
    stats.add(new EvGenesOriginStatPanel(applet));
    stats.add(new EvValueDistributionAreaStatPanel(applet));
    stats.add(new EvGenesChangeStatPanel(applet));
    stats.add(new EvDistributedObjValueStat(applet));

    radio_butt = new JRadioButton[stats.size()];

    // adding stats

    for (int i = 0; i < stats.size(); i++) {
      radio_butt[i] = new JRadioButton(stats.get(i).getName());
      radio_butt[i].addChangeListener(this);
      butt_group.add(radio_butt[i]);
      radio_butt[i].setBounds(0, 20 + 20 * i, 200, 20);
      add(radio_butt[i]);
      stats.get(i).setBounds(200, 20, 400, 200);
      add(stats.get(i));
      stats.get(i).setVisible(false);
    }

  }


  // executed when radio button state changed
  public void stateChanged(ChangeEvent e) {
    for (int i = 0; i < stats.size(); i++) {
      if (radio_butt[i].isSelected())
        stats.get(i).setVisible(true);
      else
        stats.get(i).setVisible(false);
    }

  }

}
