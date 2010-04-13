package pl.wroc.uni.ii.evolution.chart;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;

import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;

import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.bayesian.EvBayesianNetworkStatistic;

/**
 * Class containing single static method that create JPanel component with
 * wisualisation of bayesian network. Vertex are placed in circle and connected
 * with line if there is an edge.<br>
 * Vertexes that are related are painted in the same color.
 * 
 * 
 * @author Kacper Gorski (admin@34all.org)
 */
public class EvBayesianChart {

  /**
   * Disabling constructor.
   */
  protected EvBayesianChart() {
    throw new UnsupportedOperationException();
  }

  
  /**
   * 
   * Open swing component with statistics.
   * 
   * @param stats - statistics to be viewed
   */
  public static void viewStats(final EvStatistic[] stats) {
    
    final int size = 700;
    
    JFrame f = EvBayesianChart.createFrame(stats);
    
    f.pack();   
    f.setSize(size, size);
    f.show(); 
  }

  /**
   * Creates interactive JPanel component containing informations about bayesian
   * network in specified iteration.
   * Vertexes that are related are painted in the same color.
   * 
   * @param stats - statistic objects containing data about bayesian network.
   *        Each EvStatistic object represents state of bayesian network in
   *        single iteration.
   * @return JFrame component containing interactive wisualisation of the
   *         network.
   */
  public static JFrame createFrame(final EvStatistic[] stats) {

    return new BayesianNetworkWisualisation(0, stats);
  }
}

/**
 * JPanel components that wisualises vertexes and edges of bayesian network.
 * 
 * @author Kacper Gorski (admin@34all.org)
 */
class BayesianNetworkWisualisation extends JFrame implements ActionListener {

  /**
   * 
   */
  private static final long serialVersionUID = 2303838810365199956L;

  /**
   * int table of size Z x 2 storing Z edges. Each 2-elemental row represents
   * one one-way edge, first integer represent initial vertex
   */
  private int[][] edges;

  /**
   * Total number of vertexes in the network.
   */
  private int network_size = 0;

  /**
   * Color of each vertex. Each index represents corresponding edge Colors are
   * assigned randomly in the constructor.
   */
  private Color[] vertex_colors;

  
  /** 
   * tells the group number of each vertex.
   * table index = vertex, value = group
   */
  private int[] vertex_group = null;
  
  /**
   * Component used for iteration selection.
   */
  private JComboBox combo_box = new JComboBox();

  /**
   * objects storing statistics with all iterations.
   */
  private EvStatistic[] stats;

  /**
   * Iteration to be wisualised.
   */
  private int iteration;
  
  /**
   * Default constructor. Creates ready to use component wisualising data
   * specified in parameters
   * 
   * @param iteration_ - iteration to be show
   * @param stats_ - objects storing statistics
   */
  public BayesianNetworkWisualisation(final int iteration_, 
      final EvStatistic[] stats_) {
    
    final int labelWidth = 100;
    final int labelHeight = 20;
    final int comboWidth = 100;
    
    iteration = iteration_;
    
    stats = stats_;
    

    
    setLayout(null);
    
    JLabel label = new JLabel("Iteration: ");
    label.setBounds(0, 0, labelWidth, labelHeight);
    add(label);
    
    
    for (int i = 0; i < stats.length; i++) {
      combo_box.addItem(new String("" + (i + 1)));
    }    
    combo_box.setBounds(labelWidth, 0, comboWidth, labelHeight);
    add(combo_box);
    combo_box.addActionListener(this);
    

    update();

  }


  /**
   * Recalculates charts and repaint the screen.
   */
  private void update() {
    
    EvBayesianNetworkStatistic stat = 
      (EvBayesianNetworkStatistic) stats[iteration];

    
    edges = 
      stat.getNetwork();
    
    network_size = stat.getNemberOfVertex();
    
    groupColors();
    repaint();
    
  }


  /**
   * Group vertexes in building blocks and assign one color to each group.
   */
  private void groupColors() {
    
    vertex_group = new int[network_size];
    vertex_colors = new Color[network_size];

    // shuffle color for each vertex
    for (int i = 0; i < network_size; i++) {
      vertex_colors[i] = randomColor();
    }
    
    // here we store which vertexes are in group of specified index.
    ArrayList<Integer>[] groups = new ArrayList[network_size];
    for (int i = 0; i < network_size; i++) {
      vertex_group[i] = i;
      groups[i] = new ArrayList<Integer>(i);
      groups[i].add(i);
    }
    
    for (int i = 0; i < edges.length; i++) {
      int group_removed = vertex_group[edges[i][0]];
      int group_enlarged =  vertex_group[edges[i][1]];
      for (Integer vertex : groups[group_removed]) {
        vertex_group[vertex] = group_enlarged;
      }
      groups[group_enlarged].addAll(groups[group_removed]);
      groups[group_removed].clear();
    }
    
  }


  /**
   * {@inheritDoc}
   */
  @Override
  public void paint(final Graphics g) {
    super.paint(g);
    
    final int maxDegree = 360;

    // painting vertexes
    for (int i = 0; i < network_size; i++) {

      double degree = maxDegree * i / network_size;

      g.setColor(vertex_colors[vertex_group[i]]);
      drawVertex(degree, g);
      g.setColor(Color.BLACK);
      drawVertexName((i + 1) + "", degree, g);
    }

    // painting edges
    for (int i = 0; i < edges.length; i++) {
      drawEdge(edges[i][0], edges[i][1], g);
    }

  }


  /**
   * Draws vertex (small filled oval) on the oval perimeter of the component.
   * 
   * @param degree - location of the vertex on the perimeter.
   * @param g - graphics object on which vertexes are painted.
   */
  private void drawVertex(final double degree, final Graphics g) {

    final int shift = 10;
    final int vertexDiameter = 20;

    int x = getXOnArc(degree, 0);
    int y = getYOnArc(degree, 0);

    g.fillOval(x - shift, y - shift, vertexDiameter, vertexDiameter);

  }


  /**
   * Draw string on the oval perimeter of the component.
   * 
   * @param str - string to be painted.
   * @param degree - location of the string on the perimeter.
   * @param g - graphics object on which strings are painted.
   */
  private void drawVertexName(final String str, final double degree,
      final Graphics g) {

    final int shift = 25;

    int x = getXOnArc(degree, shift);
    int y = getYOnArc(degree, shift);

    g.drawString(str, x, y);
  }


  /**
   * Draws edge which starts in vertex_a and ends in vertes_b.
   * 
   * @param vertex_a - starting edge
   * @param vertex_b - ending edge
   * @param g - - graphics object on which edges are painted.
   */
  private void drawEdge(
      final int vertex_a, final int vertex_b, final Graphics g) {

    final int maxDegree = 360;

    // location of both vertexes on the perimeter
    int degree_a = maxDegree * vertex_a / network_size;
    int degree_b = maxDegree * vertex_b / network_size;

    // coordinate of both vertexes
    int x_a = getXOnArc(degree_a, 0);
    int y_a = getYOnArc(degree_a, 0);
    int x_b = getXOnArc(degree_b, 0);
    int y_b = getYOnArc(degree_b, 0);

    g.setColor(Color.BLACK);
    g.drawLine(x_a, y_a, x_b, y_b);

  }


  /**
   * Returns x coordinate of the point located on the component's oval
   * perimeter.
   * 
   * @param degree - location of the point on the perimeter.
   * @param shift - negative, shift in the direction to the center in pixels;
   *        positive, shift in the direction of the edge
   * @return x - coordinate of the point in the perimeter.
   */
  private int getXOnArc(final double degree, final int shift) {

    // shrinking perimeter so it can fit the screen
    final int shrink = 70;

    // displacement right
    final int displacement = 40;

    int radius = getHeight() / 2 - shrink;

    return displacement
        + (int) (radius + (shift + radius) * Math.cos(Math.toRadians(degree)));
  }


  /**
   * Returns y coordinate of the point located on the component's oval
   * perimeter.
   * 
   * @param degree - location of the point on the perimeter.
   * @param shift - negative, shift in the direction to the center in pixels;
   *        positive, shift in the direction of the edge
   * @return y - coordinate of the point in the perimeter.
   */
  private int getYOnArc(final double degree, final int shift) {

    // shrinking perimeter so it can fit the screen
    final int shrink = 70;
    final int topBorder = 40;

    // displacement down
    final int displacement = 40;

    int radius = getHeight() / 2 - shrink;

    return displacement
        + (int) (radius + (shift + radius) * Math.sin(Math.toRadians(degree)))
        + topBorder;
  }


  /**
   * Shuffle random color.
   * 
   * @return random color
   */
  private Color randomColor() {

    final int maxColorValue = 255;

    int red = (int) (Math.random() * maxColorValue);
    int yellow = (int) (Math.random() * maxColorValue);
    int blue = (int) (Math.random() * maxColorValue);

    return new Color(red, yellow, blue);

  }


  /**
   * {@inheritDoc}
   */
  public void actionPerformed(final ActionEvent arg0) {
    
    iteration = combo_box.getSelectedIndex();
    update();
    
  }

}
