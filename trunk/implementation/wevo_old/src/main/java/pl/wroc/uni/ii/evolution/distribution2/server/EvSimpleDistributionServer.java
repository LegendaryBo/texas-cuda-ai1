package pl.wroc.uni.ii.evolution.distribution2.server;

import java.util.logging.Logger;

import org.mortbay.jetty.Server;
import org.mortbay.jetty.servlet.ServletHandler;

/**
 * Simple server, with limited access, for simple distribution model. It wraps
 * the jetty server instance.
 * 
 * @author Karol Asgaroth Stosiek (karol.stosiek@gmail.com)
 */
public class EvSimpleDistributionServer {
  /**
   * The real jetty server instance.
   */
  private Server server;

  /**
   * Used for logging.
   */
  private final Logger logger;
  
  /**
   * Common initialization, for both constructors.
   */
  private void initServer() {
    this.logger.info("Initializing server...");
    
    ServletHandler handler = new ServletHandler();
    this.server.setHandler(handler);

    handler.addServletWithMapping(EvSimpleDistributionServlet.class, "/*");
    
    this.logger.info("Server initialized.");
  }

  /**
   * Construction with port parameter.
   * 
   * @param port - port number, on which the server listens to.
   */
  public EvSimpleDistributionServer(final int port) {
    if (port < 0) {
      throw new IllegalArgumentException("Port number must not be negative!");
    }

    /* since the logger field is final, we cannot 
     * place these two lines in the init() method */
    this.logger = Logger.getLogger(
        EvSimpleDistributionServer.class.getCanonicalName());
    
    this.server = new Server(port);
    initServer();
  }

  /**
   * Constructor with default values.
   */
  public EvSimpleDistributionServer() {
    
    /* since the logger field is final, we cannot 
     * place these two lines in the init() method */
    this.logger = Logger.getLogger(
        EvSimpleDistributionServer.class.getCanonicalName());
    
    this.server = new Server();
    initServer();
  }

  /**
   * Starting the server.
   * 
   * @throws Exception - thrown, when there's a problem with jetty server
   *         instance.
   */
  public void run() throws Exception {
    this.logger.info("Starting server...");
    this.server.start();
    this.logger.info("Server started.");
  }

  /**
   * Stopping the evolution server.
   * 
   * @throws Exception - thrown, when there's a problem with jetty server
   *         instance.
   */
  public void stop() throws Exception {
    this.logger.info("Stopping server...");
    this.server.stop();
    this.logger.info("Server stopped.");
  }
}
