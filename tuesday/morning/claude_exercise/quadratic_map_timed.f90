program quadratic_map_timed
    implicit none
    integer, parameter :: dp = kind(1.0d0)
    real(dp), allocatable :: x_vals(:), g_vals(:)
    real(dp), allocatable :: trajectory(:)
    real(dp), allocatable :: histogram_data(:)
    integer :: num_points, traj_n, hist_n
    real(dp) :: start_time, end_time

    ! Set sizes
    num_points = 1000
    traj_n = 150
    hist_n = 100000

    ! Allocate arrays
    allocate(x_vals(0:num_points))
    allocate(g_vals(0:num_points))
    allocate(trajectory(0:traj_n))
    allocate(histogram_data(0:hist_n))

    ! START TIMING
    call cpu_time(start_time)

    ! Generate all data
    call generate_function_data(x_vals, g_vals, num_points)
    call generate_trajectory_data(0.3_dp, traj_n, trajectory)
    call generate_histogram_array(0.3_dp, hist_n, histogram_data)

    ! END TIMING
    call cpu_time(end_time)

    print '(A,F12.6,A)', 'Fortran data generation time: ', end_time - start_time, ' seconds'

    ! Cleanup
    deallocate(x_vals, g_vals, trajectory, histogram_data)

contains

    ! Quadratic map function
    real(dp) function quadratic_map_func(x)
        real(dp), intent(in) :: x
        quadratic_map_func = 4.0_dp * x * (1.0_dp - x)
    end function quadratic_map_func

    ! Generate function data
    subroutine generate_function_data(x_vals, g_vals, num_points)
        real(dp), intent(out) :: x_vals(0:)
        real(dp), intent(out) :: g_vals(0:)
        integer, intent(in) :: num_points
        integer :: i

        do i = 0, num_points
            x_vals(i) = real(i, dp) / real(num_points, dp)
            g_vals(i) = quadratic_map_func(x_vals(i))
        end do
    end subroutine generate_function_data

    ! Generate trajectory data
    subroutine generate_trajectory_data(x0, n, trajectory)
        real(dp), intent(in) :: x0
        integer, intent(in) :: n
        real(dp), intent(out) :: trajectory(0:)
        real(dp) :: x
        integer :: t

        x = x0
        do t = 0, n
            trajectory(t) = x
            x = quadratic_map_func(x)
        end do
    end subroutine generate_trajectory_data

    ! Generate histogram data
    subroutine generate_histogram_array(x0, n, histogram_data)
        real(dp), intent(in) :: x0
        integer, intent(in) :: n
        real(dp), intent(out) :: histogram_data(0:)
        real(dp) :: x
        integer :: t

        x = x0
        do t = 0, n
            histogram_data(t) = x
            x = quadratic_map_func(x)
        end do
    end subroutine generate_histogram_array

end program quadratic_map_timed
