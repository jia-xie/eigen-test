#include <iostream>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <complex>
#define N_STATES (3)
#define N_INPUTS (1)

int main()
{
        Eigen::MatrixXd A(3, 3);
        Eigen::MatrixXd B(3, 1);
        Eigen::MatrixXd Q(3, 3);
        Eigen::MatrixXd R(1, 1);

        A << 1, -0.0303954624728227, -0.000759886561820567,
            0, 1.01190488946852, 0.0502976222367131,
            0, 0.476195578740889, 1.01190488946852;
        B << 8.0471,
            -0.8011,
            -0.0438;

        Q << 1, 0, 0,
            0, 1, 0,
            0, 0, 1;
        R << 1;

        Eigen::MatrixXd B_R_inv_Bt = B * R.inverse() * B.transpose();
        std::cout << "B_R_inv_Bt:\n"
                  << B_R_inv_Bt << std::endl;
        Eigen::MatrixXd A_t_inv = A.transpose().inverse();
        std::cout << "A_t_inv:\n"
                  << A_t_inv << std::endl;

        // Calculate the components of the Hamiltonian matrix
        Eigen::MatrixXd H11 = A + B_R_inv_Bt * A_t_inv * Q;
        std::cout << "H11:\n"
                  << H11 << std::endl;
        Eigen::MatrixXd H12 = -B_R_inv_Bt * A_t_inv;
        std::cout << "H12:\n"
                  << H12 << std::endl;
        Eigen::MatrixXd H21 = -A_t_inv * Q;
        std::cout << "H21:\n"
                  << H21 << std::endl;
        Eigen::MatrixXd H22 = A_t_inv;
        std::cout << "H22:\n"
                  << H22 << std::endl;

        // Assemble the Hamiltonian matrix
        Eigen::MatrixXd HamiltonianMatrix(A.rows() * 2, A.cols() * 2);
        HamiltonianMatrix << H11, H12,
            H21, H22;

        // Solve the eigenproblem for the Hamiltonian matrix
        Eigen::EigenSolver<Eigen::MatrixXd> eigen_solver(HamiltonianMatrix);
        std::cout << "Eigenvalues:\n"
                  << eigen_solver.eigenvalues() << std::endl;
        std::cout << "Eigenvectors:\n"
                  << eigen_solver.eigenvectors() << std::endl;

        Eigen::MatrixXd stable_eigen_vectors(N_STATES * 2, N_STATES);
        int stable_eigen_counter = 0;
        std::vector<int> stable_eigen_indices = {};
        // Iterate through eigenvalues and rearrange eigenvectors
        for (int i = 0; i < eigen_solver.eigenvalues().size(); i++)
        {
                if (stable_eigen_counter == A.rows())
                {
                        break;
                }
                // Check if eigenvalue length or norm is less than 1
                if (std::abs(eigen_solver.eigenvalues()[i]) < 1)
                {
                        stable_eigen_counter++;
                        // Get corresponding eigenvector
                        Eigen::VectorXcd eigenvector = eigen_solver.eigenvectors().col(i);
                        // stable_eigen_vectors << eigenvector;
                        stable_eigen_indices.push_back(i);
                        // Print the rearranged eigenvector
                        std::cout << "Rearranged Eigenvector " << i << ":\n"
                                  << eigenvector << std::endl;
                }
        }
        stable_eigen_vectors << eigen_solver.eigenvectors().col(stable_eigen_indices[0]).real(), eigen_solver.eigenvectors().col(stable_eigen_indices[1]).real(), eigen_solver.eigenvectors().col(stable_eigen_indices[2]).real();
        std::cout << "Stable Eigenvectors:\n"
                  << stable_eigen_vectors << std::endl;

        // Create the upper and lower matrices
        Eigen::MatrixXd upper_matrix = stable_eigen_vectors.topRows(N_STATES);
        Eigen::MatrixXd lower_matrix = stable_eigen_vectors.bottomRows(N_STATES);

        // Output the upper matrix
        std::cout << "Upper matrix:" << std::endl;
        std::cout << upper_matrix << std::endl
                  << std::endl;

        // Output the lower matrix
        std::cout << "Lower matrix:" << std::endl;
        std::cout << lower_matrix << std::endl;
        return 0;
}
