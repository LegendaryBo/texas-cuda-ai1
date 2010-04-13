package pl.wroc.uni.ii.evolution.distribution2;

import pl.wroc.uni.ii.evolution.distribution2.server.EvSimpleDistributionServer;
import junit.framework.TestCase;

/**
 * Tests EvDistributionServer constructor.
 * 
 * @author Karol Asgaroth Stosiek (karol.stosiek@gmail.com)
 */
public class EvSimpleDistributionServerTest extends TestCase {
  
  /**
   * Tests constructor with parameter.
   */
  public void testConstructorWithPortParameter() {
    
    @SuppressWarnings("unused")
    EvSimpleDistributionServer server;
    
    try {
      server = new EvSimpleDistributionServer(-1);
    } catch (Exception e) {
      /* this is expected to happen */
      assertTrue(true);
    }
    
    try {
      server = new EvSimpleDistributionServer(8080);
    } catch (Exception e) {
      /* this is not expected to happen */
      assertTrue(false);
    }
    
  }
}
