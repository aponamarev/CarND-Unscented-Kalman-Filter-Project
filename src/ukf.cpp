#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include <math.h>
#define _USE_MATH_DEFINES
#define EPS 0.0001

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
    is_initialized_ = false;
    
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;
    ld = 2; // Set a parameter for lidar output dimentionality
    
    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true ;
    rd = 3; // Set a parameter for radar output dimentionality
    
    // initial state vector
    x_ = VectorXd(5);
    x_ << 1,1,0,0,0;
    
    // initial covariance matrix
    P_ = MatrixXd(5, 5);
    P_ <<
    0.09,0,0,0,0,
    0,0.09,0,0,0,
    0,0,1,0,0,
    0,0,0,1,0,
    0,0,0,0,1;
    
    
    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 3; // max acceleration in the city 6 m/s^2, std is a half
    
    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = M_PI/6; // max turning angle is probably aroud 60 degrees (pi/3), std is a half
    
    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;
    
    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;
    
    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;
    
    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;
    
    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;
    
    n_x_ = 5;
    n_aug_ = 7;
    lambda_ = 3 - n_aug_;
    
    Xsig_pred_ = MatrixXd(n_x_, n_aug_*2+1);
    
    n_aud_sig_points = int(Xsig_pred_.cols());
    
    weights_ = VectorXd(n_aud_sig_points);
    weights_(0) = lambda_ / (lambda_ + n_aug_);
    for (int i=1; i<n_aud_sig_points;i++) {
        weights_(i) = 0.5 / (lambda_ + n_aug_);
    }
    // Set a measurement covariance/uncertainty matrix for lidar
    R_l = MatrixXd(ld,ld);
    R_l <<
    std_laspx_*std_laspx_, 0,
    0, std_laspy_ * std_laspy_;
    // Set a measurement covariance/uncertainty matrix for radar
    R_r = MatrixXd(rd,rd);
    R_r <<
    std_radr_ * std_radr_, 0, 0,
    0, std_radphi_ * std_radphi_, 0,
    0, 0, std_radrd_ * std_radrd_;
}

void UKF::normAngle(double &phi) {
    phi = atan2(sin(phi), cos(phi));
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
    
    if (!is_initialized_) {
        // Initialize time stamp
        time_us_ = meas_package.timestamp_;
        if (use_laser_ && meas_package.sensor_type_==MeasurementPackage::LASER) {
            x_(0) = meas_package.raw_measurements_(0);
            x_(1) = meas_package.raw_measurements_(1);
            if (fabs(x_(0)) < EPS and fabs(x_(1)) < EPS) {
                x_(0) = EPS;
                x_(1) = EPS;
            }
        }
        else if (use_radar_ && meas_package.sensor_type_==MeasurementPackage::RADAR) {
            double rho = meas_package.raw_measurements_(0);
            double phi = meas_package.raw_measurements_(1);
            double rho_dot = meas_package.raw_measurements_(2);
            double vx = cos(phi) * rho_dot;
            double vy = sin(phi) * rho_dot;
            x_(0) = cos(phi) * rho;
            x_(1) = sin(phi) * rho;
            x_(2) = sqrt(vx * vx + vy * vy);
        }
        is_initialized_ = true;
        return;
    }
    
    // compute time difference
    double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
    time_us_ = meas_package.timestamp_;
    
    Prediction(dt);
    
    if (meas_package.sensor_type_==MeasurementPackage::LASER) {
        UpdateLidar(meas_package);
    } else {
        UpdateRadar(meas_package);
    }
    
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double dt) {
    
    // ***************************************
    // **** Create augmented sigma points ****
    // ***************************************
    double dt2 = dt*dt;
    // 1. Create augmented x_aug
    VectorXd x_aug = VectorXd(n_aug_);
    x_aug.fill(0.0);
    x_aug.head(n_x_) = x_;
    // 2. Create a matrix for augmented x_sig_aug
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2*n_aug_+1);
    // 3. Create P_aug
    MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
    P_aug.fill(0.0);
    // 4. Calculate P_aug
    P_aug.topLeftCorner(n_x_, n_x_) = P_;
    P_aug(n_x_,n_x_) = std_a_*std_a_;
    P_aug(n_x_+1,n_x_+1) = std_yawdd_*std_yawdd_;
    // 5. Calculate A matrix = sqrt((lambda + n_aug)*P_aug)
    MatrixXd A_aug = ((lambda_+n_aug_)*P_aug).llt().matrixL();
    Xsig_aug.col(0) = x_aug;
    for (int i=0; i<n_aug_; i++) {
        Xsig_aug.col(i+1) = x_aug + A_aug.col(i);
        Xsig_aug.col(i+1+n_aug_) = x_aug - A_aug.col(i);
    }
    
    // ****************************************
    // **** Predict augmented sigma points ****
    // ****************************************
    for (int i=0; i < n_aud_sig_points; i++) {
        double px = Xsig_aug(0,i);
        double py = Xsig_aug(1,i);
        double v = Xsig_aug(2,i);
        double yaw = Xsig_aug(3,i);
        double yaw_dot = Xsig_aug(4,i);
        double v_a = Xsig_aug(5,i);
        double v_yaw_dot = Xsig_aug(6,i);
        const double yaw_1 = yaw + dt*yaw_dot;
        double sin_yaw = sin(yaw);
        double cos_yaw = cos(yaw);
        //avoid division by zero
        if (fabs(yaw_dot)>EPS) {
            double v_yaw_dot = v/yaw_dot;
            //write predicted sigma points into right column
            px += v_yaw_dot * ( sin(yaw_1) - sin_yaw )
            + 0.5 * dt2 * cos_yaw * v_a;
            py += v_yaw_dot * ( -cos(yaw_1) + cos_yaw )
            + 0.5 * dt2 * sin_yaw * v_a;
        } else {
            px += dt * cos_yaw * v + 0.5 * dt2 * v_a * cos_yaw;
            py += dt * sin_yaw * v + 0.5 * dt2 * v_a * sin_yaw;
        }
        v += dt*v_a;
        yaw += dt*yaw_dot + 0.5 * dt2 * v_yaw_dot;
        yaw_dot += dt * v_yaw_dot;
        
        Xsig_pred_.col(i) << px, py, v, yaw, yaw_dot;
    }
    
    // ********************************
    // **** Calculate mean x and P ****
    // ********************************
    x_ = Xsig_pred_ * weights_;
    P_.fill(0.0);
    for (int i = 0; i < n_aud_sig_points; i++) {
        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        //angle normalization
        normAngle(x_diff(3));
        P_ += weights_(i) * x_diff * x_diff.transpose();
    }
}

void UKF::ukfUpdate(const MatrixXd &S, MatrixXd &Z_p, const VectorXd &z, const VectorXd &z_mean) {
    const int ndim = int(z.size());
    // 5. calculate cross correlation matrix
    MatrixXd Tc = MatrixXd(n_x_, ndim);
    Tc.fill(0.0);
    for (int i = 0; i < n_aud_sig_points; i++) {
        // 5.1 residual
        VectorXd z_diff = Z_p.col(i) - z_mean;
        normAngle(z_diff(1));
        // 5.2 state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        normAngle(x_diff(3));
        Tc += weights_(i) * x_diff * z_diff.transpose();
    }
    // 6. Kalman gain K
    MatrixXd K = Tc * S.inverse();
    // 6.1 residual
    VectorXd z_diff = z - z_mean;
    normAngle(z_diff(1));
    // 6.2 update state mean and covariance matrix
    x_ += K * z_diff;
    P_ -= K*S*K.transpose();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
    /**
     TODO:
     
     You'll also need to calculate the lidar NIS.
     */
    
    /*
     ******************************************
     **** z_mean = mean(z)                 ****
     **** S = w*(z-z_mean)*(z-z_mean).T + R****
     ******************************************
     */
    VectorXd z = meas_package.raw_measurements_.head(ld);
    // 1. Create Z_p matrix that will capture transformation of Xsig points
    MatrixXd Z_p = Xsig_pred_.block(0, 0, ld, n_aud_sig_points);
    // 2. Convert Xsigma points to measurement space
    // z(k+1) = h(x(k+1)) + w
    // 3. Calculate Z prediction mean
    VectorXd z_mean = Z_p * weights_;
    // 4. Calculate total covariance (uncertainty)
    // S = w*H + R where H = delta_Z * delta_z.T
    MatrixXd S = MatrixXd(ld,ld);
    S.fill(0.0);
    for (int i=0; i < n_aud_sig_points; i++) {
        VectorXd delta = Z_p.col(i) - z_mean;
        S += weights_(i) * delta * delta.transpose();
    }
    S += R_l;
    
    ukfUpdate(S, Z_p, z, z_mean);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
    /**
     TODO:
     
     You'll also need to calculate the lidar NIS.
     */
    
    /*
     ******************************************
     **** z_mean = mean(z)                 ****
     **** S = w*(z-z_mean)*(z-z_mean).T + R****
     ******************************************
     */
    VectorXd z = meas_package.raw_measurements_.head(rd);
    
    // 1. Create Z_p matrix that will capture transformation of Xsig points
    MatrixXd Z_p = MatrixXd(rd, n_aud_sig_points);
    // 2. Convert Xsigma points to measurement space
    // z(k+1) = h(x(k+1)) + w
    for (int i=0; i < n_aud_sig_points; i++) {
        double px = Xsig_pred_(0,i);
        double py = Xsig_pred_(1,i);
        double yaw = Xsig_pred_(3,i);
        
        double v_x = cos(yaw)*Xsig_pred_(2,i);
        double v_y = sin(yaw)*Xsig_pred_(2,i);
        double rho = sqrt(px*px + py*py);
        
        // measurement model
        Z_p(0,i) = rho; //rho
        Z_p(1,i) = atan2(py,px); //yaw
        Z_p(2,i) = (px*v_x + py*v_y) / rho; //r_dot
    }
    // 3. Calculate Z prediction mean
    VectorXd z_mean = Z_p * weights_;
    // 4. Calculate total covariance (uncertainty)
    // S = w*H + R where H = delta_Z * delta_z.T
    MatrixXd S = MatrixXd(rd, rd);
    S.fill(0.0);
    for (int i=0; i < n_aud_sig_points; i++) {
        VectorXd delta = Z_p.col(i) - z_mean;
        normAngle(delta(1));
        S += weights_(i) * delta * delta.transpose();
    }
    S += R_r;
    
    // 5. calculate cross correlation matrix
    ukfUpdate(S, Z_p, z, z_mean);
}
