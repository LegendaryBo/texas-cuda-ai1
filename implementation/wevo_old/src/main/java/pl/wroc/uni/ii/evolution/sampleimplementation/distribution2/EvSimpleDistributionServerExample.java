package pl.wroc.uni.ii.evolution.sampleimplementation.distribution2;

import pl.wroc.uni.ii.evolution.distribution2.server.EvSimpleDistributionServer;

/**
 * Simple example of a running server in simple distribution model.
 * 
 * @author Karol Asgaroth Stosiek (karol.stosiek@gmail.com)
 */
public final class EvSimpleDistributionServerExample {

  /**
   * This construtor disables the generation of public 
   * default constructor in java class file.
   */
  private EvSimpleDistributionServerExample() {
  }
  
  /**
   * Static method to run the example.
   * 
   * @param args - ignored.
   */
  public static void main(final String[] args) {
    EvSimpleDistributionServer server =
      new EvSimpleDistributionServer(8080);

    try {
      server.run();
    } catch (Exception e) {
      e.printStackTrace();
    }
  }
}
